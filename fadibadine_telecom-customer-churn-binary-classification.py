import os

import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))



from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import AUC, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Recall, Precision

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
init = {

    "datadir": "/kaggle/input/telco-customer-churn/",

    "datafile": "WA_Fn-UseC_-Telco-Customer-Churn.csv",

    "test_split": 0.1,

    "test_random_state": 42,

    "val_split": 0.1,

    "val_random_state": 43,

    "clear_logs": True,

    "classweights": True

}
def load_telco_churn_data(dataset_path, dataset_file):

    assert(dataset_path is not None and dataset_file is not None)

    

    csv_path = os.path.join(dataset_path, dataset_file)

    

    return pd.read_csv(csv_path)



def plot_feature(df,

                  x,

                  y = "Percent",

                  title = None,

                  x_label = None,

                  y_label = None,

                  x_axis_split = None,

                  figsize=(16,5),

                  annotate = False,

                  sort_per_feature = True,

                  exclude_values = None):

    fig = plt.figure(figsize=figsize)



    if exclude_values is not None and isinstance(exclude_values, str):

        feature_counts = (df.loc[df[x] != exclude_values].groupby(['Churn'])[x]

                        .value_counts(normalize=True)

                        .rename(y)

                        .mul(100)

                        .reset_index())

    else:

        feature_counts = (df.groupby(['Churn'])[x]

                        .value_counts(normalize=True)

                        .rename(y)

                        .mul(100)

                        .reset_index())

        

    if sort_per_feature is True:

        feature_counts = feature_counts.sort_values(x, ascending=False)

    

    p = sns.barplot(x=x, y=y, hue="Churn", data=feature_counts)

    if title is not None:

        p.set_title(title, fontsize=20)

    if x_label is not None:

        p.set_xlabel(x_label)

    if y_label is not None:

        p.set_ylabel(y_label)



    if x_axis_split is not None:

        _, _ = plt.xticks(np.arange(df[x].min(),

                             df[x].max(),

                             x_axis_split),

                          np.arange(df[x].min(),

                             df[x].max(),

                             x_axis_split))

    if annotate is True:

        sizes = []

        for patch in p.patches:

            h, w = patch.get_height(), patch.get_width()

            sizes.append(h)



            p.annotate(format(h, '.2f'),

                       (patch.get_x() + w / 2., h),

                       ha = 'center',

                       va = 'center',

                       xytext = (0, 10),

                       textcoords = 'offset points')

            

        p.set_ylim(0, max(sizes) * 1.15)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Reading the dataset and printing its header

df_data = load_telco_churn_data(dirname, filename)

print("Data dimension: {}".format(df_data.shape))



# Create a train / test split

df_train, df_test = train_test_split(df_data,

                                       test_size=init["test_split"],

                                       random_state=init["test_random_state"])



print("Train Data dimension: {}".format(df_train.shape))

print("Test Data dimension: {}".format(df_test.shape))

df_train.head()
no_yes_percentage = df_train["Churn"].value_counts() / df_train.shape[0] * 100



print("In the training set, {0:.2f}% will churn and {1:.2f}% will not".format(no_yes_percentage[1], no_yes_percentage[0]))
print(df_train.info())
print(df_train.isnull().sum())
print(df_train["TotalCharges"].to_frame().head())
print("There are {} white space occurrences in TotalCharges".format(df_train["TotalCharges"].str.count(" ").sum()))
o_shape = df_train.shape



df_train.drop(df_train.loc[df_train["TotalCharges"] == " "].index, inplace=True)



print("Shape has been reduced from {} to {}".format(o_shape, df_train.shape))
plot_feature(df_train,

              x = "gender",

              title = "Churn Percentage / Gender",

              x_label = "Gender",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

              exclude_values = 'test')
plot_feature(df_train,

              x = "SeniorCitizen",

              title = "Churn Percentage / Senior Citizen",

              x_label = "Senior Citizen",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)
plot_feature(df_train,

              x = "Partner",

              title = "Churn Percentage / Partner",

              x_label = "Partner",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)



plot_feature(df_train,

              x = "Dependents",

              title = "Churn Percentage / Dependents",

              x_label = "Dependents",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)
plot_feature(df_train,

              x = "PhoneService",

              title = "Churn Percentage / Phone Service",

              x_label = "Phone Service",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)



plot_feature(df_train,

              x = "MultipleLines",

              title = "Churn Percentage / Multiple Lines",

              x_label = "Multiple Lines",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

              exclude_values = "No phone service")
plot_feature(df_train,

              x = "InternetService",

              title = "Churn Percentage / Internet Service",

              x_label = "Internet Service",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)
plot_feature(df_train,

              x = "OnlineSecurity",

              title = "Churn Percentage / Online Security",

              x_label = "Online Security",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")



plot_feature(df_train,

              x = "OnlineBackup",

              title = "Churn Percentage / Online Backup",

              x_label = "Online Backup",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")
plot_feature(df_train,

              x = "DeviceProtection",

              title = "Churn Percentage / Device Protection",

              x_label = "Device Protection",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")



plot_feature(df_train,

              x = "TechSupport",

              title = "Churn Percentage / Tech Support",

              x_label = "Tech Support",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")

plot_feature(df_train,

              x = "StreamingTV",

              title = "Churn Percentage / Streaming TV",

              x_label = "Streaming TV",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")



plot_feature(df_train,

              x = "StreamingMovies",

              title = "Churn Percentage / Streaming Movies",

              x_label = "Streaming Movies",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True,

             exclude_values="No internet service")
plot_feature(df_train,

              x = "Contract",

              title = "Churn Percentage / Contract",

              x_label = "Contract",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)
plot_feature(df_train,

              x = "tenure",

              title = "Churn Percentage / Tenure",

              x_label = "Tenure (months)",

              y_label = "Percentage",

              x_axis_split = 5)
plot_feature(df_train,

              x = "PaperlessBilling",

              title = "Churn Percentage / Paperless Billing",

              x_label = "Paperless Billing",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)



plot_feature(df_train,

              x = "PaymentMethod",

              title = "Churn Percentage / Payment Method",

              x_label = "Payment Method",

              y_label = "Percentage",

              x_axis_split = None,

              annotate = True)
col     = "MonthlyCharges"

sep     = " "

suf     = "Ranges"

new_col = col + sep + suf



bins    = 30

x_min   = df_train[col].min()

x_max   = df_train[col].max()

x_step  = (x_max - x_min) / bins



df_train[new_col] = pd.cut(df_train[col], bins, right=True, labels=False)

df_train.loc[:, [new_col]] = ((df_train[new_col] + 1) * x_step + x_min).astype(np.int32)



plot_feature(df_train,

              x = new_col,

              title = "Churn Percentage / Monthly Charges",

              x_label = "Monthly Charges Ranges (less than)",

              y_label = "Percentage",

              x_axis_split = None,

              sort_per_feature=False)



df_train = df_train.drop(columns=[new_col])
col     = "TotalCharges"

sep     = " "

suf     = "Ranges"

new_col = col + sep + suf



df_train[col] = df_train[col].map(lambda x: np.nan if x in [' '] else np.float64(x))



bins    = 30

x_min   = df_train[col].min()

x_max   = df_train[col].max()

x_step  = (x_max - x_min) / bins



df_train[new_col] = pd.cut(df_train[col], bins, right=True, labels=False)

df_train.loc[:, [new_col]] = ((df_train[new_col] + 1) * x_step + x_min).astype(np.int32)



plot_feature(df_train,

              x = new_col,

              title = "Churn Percentage / Total Charges",

              x_label = "Total Charges Ranges (less than)",

              y_label = "Percentage",

              x_axis_split = None,

              sort_per_feature=False)



df_train = df_train.drop(columns=[new_col])
# If df_train and df_test were already read, delete them in order

# to load them again and let the automatic data preprocessing take

# care of the wrong data in TotalAmounts column automatically

try:

    del df_train

    del df_test

except NameError:

    pass



df_data = load_telco_churn_data(init["datadir"], init["datafile"])



# Create a (train, val) / test split

df_train, df_test = train_test_split(df_data,

                                       test_size=init["test_split"],

                                       random_state=init["test_random_state"])





# Copy in order not to have a warning when trying to alter the data

df_train_features = df_train.copy()

df_train_labels   = df_train["Churn"].map(dict(Yes=1, No=0)).copy()

df_train_features = df_train.drop("Churn", axis=1)



df_test_features  = df_test.copy()

df_test_labels    = df_test["Churn"].map(dict(Yes=1, No=0)).copy()

df_test_features  = df_test.drop("Churn", axis=1)



# Remove df_data

del df_data, df_train, df_test
class ObjectToFloatEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, replace_value = np.nan):

        self.replace_value = replace_value

    

    def fit(self, X, y = None):

        return self

    

    def transform(self, X, y = None):

        if isinstance(X, pd.DataFrame):

            for s_name in list(X):

                X[s_name] = X[s_name].map(lambda x: self.replace_value if not self._is_float(x) else np.float64(x))

        elif isinstance(X, pd.Series):

            X = X.map(lambda x: self.replace_value if not self._is_float(x) else np.float64(x))

        else:

            X = np.apply_along_axis(self._to_float64, 0, X)

        

        return X



    def _is_float(self, s):

        try:

            np.float64(s)

            return True

        except ValueError:

            return False



    def _to_float64(self, s):

        try:

            f = np.float64(s)

        except ValueError:

            f = [np.float64(x) if is_float(x) else self.replace_value for x in s]

        

        return f

 
## Features that do not need any changes => Passthrough

pass_attrs = ["SeniorCitizen"]



pass_pipeline = Pipeline([

    ('pass', "passthrough")

])





## Features that are encoded as ordinal values

ordinal_attrs = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]



ordinal_pipeline = Pipeline([

    ('ordinal', OrdinalEncoder()),

    ('std_scaler', StandardScaler())

])





## OneHot Encoded features

onehot_attrs = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",

                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",

                "Contract", "PaymentMethod"]



onehot_pipeline = Pipeline([

    ('onehot', OneHotEncoder())

])



## TotalCharges is encoded as object while we need it as a float

## hence we apply the already defined ObjectToFloatEncoder

total_attrs = ["TotalCharges"]



total_pipeline = Pipeline([

    ('obj_to_float', ObjectToFloatEncoder()),

    ('imputer', SimpleImputer(strategy="mean")),

    ('std_scaler', StandardScaler())

])



## Float features

num_attrs = ["tenure", "MonthlyCharges"]



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('std_scaler', StandardScaler())

])



## The full Pipeline

full_pipeline = ColumnTransformer([

    ("pass", pass_pipeline, pass_attrs),

    ("ordinal", ordinal_pipeline, ordinal_attrs),

    ("onehot", onehot_pipeline, onehot_attrs),

    ("total", total_pipeline, total_attrs),

    ("num", num_pipeline, num_attrs)

], remainder='drop')
df_train_processed = full_pipeline.fit_transform(df_train_features)

df_test_processed  = full_pipeline.fit_transform(df_test_features)



print("Dataset dimension transformed from {} to {}".format(df_train_features.shape[1], df_train_processed.shape[1]))



print("Training set dimension: {}".format(df_train_processed.shape))

print("Test set dimension: {}".format(df_test_processed.shape))
counts = np.bincount(df_train_labels)

print(

    "Number of positive samples in training data: {} ({:.2f}% of total)".format(

        counts[1], 100 * float(counts[1]) / df_train_labels.shape[0]

    )

)



class_weights = {0: 1.0 / counts[0], 1: 1.0 / counts[1]}

print(class_weights)
tf.keras.backend.clear_session()



np.random.seed(42)

tf.random.set_seed(42)
# TPU detection  

try:

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

except ValueError:

  tpu = None



# TPUStrategy for distributed training

if tpu:

  tf.config.experimental_connect_to_cluster(tpu)

  tf.tpu.experimental.initialize_tpu_system(tpu)

  strategy = tf.distribute.experimental.TPUStrategy(tpu)

else: # default strategy that works on CPU and single GPU

  strategy = tf.distribute.get_strategy()



print(strategy)
def build_model(input_shape, strategy, metrics):

    with strategy.scope():

        inputs = Input(shape=input_shape, name="input")



        x = inputs



        x = Dense(20, activation='relu', name="dense1")(x)

        x = Dropout(0.2, name="dropout1")(x)

        x = Dense(15, activation='relu', name="dense2")(x)

        x = Dropout(0.4, name="dropout2")(x)

        x = Dense(20, activation='relu', name="dense3")(x)

        x = Dropout(0.2, name="dropout3")(x)

        x = Dense(25, activation='relu', name="dense4")(x)

        x = Dropout(0.3, name="dropout4")(x)



        outputs = Dense(1, activation='sigmoid', name="output")(x)



        model = Model(inputs, outputs)



        model.compile(optimizer=Adam(lr=0.0035157669392935006), # 0.0035157669392935006

                                    loss='binary_crossentropy',

                                    metrics=metrics)



    return model



metrics = [

    FalseNegatives(name="fn"),

    FalsePositives(name="fp"),

    TrueNegatives(name="tn"),

    TruePositives(name="tp"),

    Precision(name="precision"),

    Recall(name="recall"),

    AUC(name="auc")

]



model = build_model(df_train_processed.shape[1:], strategy=strategy, metrics=metrics)

model.summary()
earlystopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

mdlcheckpoint_cb = ModelCheckpoint("model.h5", monitor="val_fn", save_best_only=True)
history = model.fit(df_train_processed, df_train_labels, epochs=100,

                validation_split=init["val_split"],

                class_weight=class_weights if init["classweights"] is True else None,

                callbacks=[earlystopping_cb, mdlcheckpoint_cb])
best_model = tf.keras.models.load_model("model.h5")
print(best_model.evaluate(df_test_processed, df_test_labels))