# libraries
import pandas as pd
import pandas_profiling as pp
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column as fc
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_curve
# data
heart_df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
heart_df.head()
# pp.ProfileReport(heart_df) # takes up a bit of space
# train test split
train_df, test_df = train_test_split(heart_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)
features = {
    "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
    "categorical": ["sex", "cp", "fbs", "restecg", "exang", "thal"],
}
# get categorical values
categorical_values = {}
for f in features["categorical"]:
    categorical_values[f] = train_df[f].unique()
# convert probability to class
def categorize(x: float):
    return 0 if x < 0.5 else 1
# fit a simple logistic regression
transformations = {
    "numeric": make_pipeline(StandardScaler()),
    "categorical": make_pipeline(OneHotEncoder())
}

preprocessor = ColumnTransformer([
    ("numeric", transformations["numeric"], features["numeric"]),
    ("categorical", transformations["categorical"], features["categorical"]),
])

baseline_model = make_pipeline(preprocessor, LogisticRegression())

baseline_model.fit(train_df, train_df.target)
baseline_preds = baseline_model.predict_proba(val_df)[:, 1]
baseline_accuracy = accuracy_score(val_df.target, list(map(categorize, baseline_preds)))
baseline_fpr, baseline_tpr, _ = roc_curve(val_df.target, baseline_preds)
# convert dataframe to dataset
def df_to_ds(df: pd.DataFrame, batch_size: int = 50) -> tf.data.Dataset:
    df_copy = df.copy()
    labels = df_copy.pop("target")
    return (
        tf.data.Dataset
        .from_tensor_slices((dict(df_copy), labels))
        .batch(batch_size)
    )

ds_dict = {}
for kind, data in [("train", train_df), ("test", test_df), ("val", val_df)]:
    ds_dict[kind] = df_to_ds(data)
def build_feature_layer(features: dict, categorical_values: dict):
    
    layer_content = []
    feature_collection = {}
    
    # format conversion
    for f in features["numeric"] + features["categorical"]:
        feature_collection[f] = fc.numeric_column(f)
        
    # just append numeric columns
    for f in features["numeric"]:
        layer_content.append(feature_collection[f])
        
    # encode categorical
    for f in features["categorical"]:
        one_hot = fc.categorical_column_with_vocabulary_list(f, categorical_values[f])
        layer_content.append(fc.indicator_column(one_hot))
        
    # extras
    age_buckets = fc.bucketized_column(feature_collection["age"], [20,30,40,50,60,70])
    layer_content.append(age_buckets)
    
    return layers.DenseFeatures(layer_content)

input_features = build_feature_layer(features, categorical_values)
model = tf.keras.Sequential()
model.add(input_features)
model.add(layers.Dense(128))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(ds_dict["train"], validation_data=ds_dict["test"], epochs=100)
nn_preds = tf.squeeze(model.predict(ds_dict["val"])).numpy()
nn_accuracy = accuracy_score(val_df.target, list(map(categorize, nn_preds)))
nn_fpr, nn_tpr, _ = roc_curve(val_df.target, nn_preds)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(baseline_fpr, baseline_tpr, label=f'Baseline (accuracy: {int(baseline_accuracy*100)}%)')
plt.plot(nn_fpr, nn_tpr, label=f'NN (accuracy: {int(nn_accuracy*100)}%)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
