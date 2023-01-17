import pandas as pd

# Some sklearn tools for preprocessing and building a pipeline. 

# ColumnTransformer was introduced in 0.20 so make sure you have this version

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report



# Our algorithms, by from the easiest to the hardest to intepret.

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
df = pd.read_csv('../input/bank-additional-full.csv', sep = ';')
df.y.value_counts()
# Get X, y

y = df["y"].map({"no":0, "yes":1})

X = df.drop("y", axis=1)
X.drop("duration", inplace=True, axis=1)
X.dtypes
# Some such as default would be binary features, but since

# they have a third class "unknown" we'll process them as non binary categorical

num_features = ["age", "campaign", "pdays", "previous", "emp.var.rate", 

                "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]



cat_features = ["job", "marital", "education","default", "housing", "loan",

                "contact", "month", "day_of_week", "poutcome"]
preprocessor = ColumnTransformer([("numerical", "passthrough", num_features), 

                                  ("categorical", OneHotEncoder(sparse=False, handle_unknown="ignore"),

                                   cat_features)])
# Logistic Regression

lr_model = Pipeline([("preprocessor", preprocessor), 

                     ("model", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42))])



# Decision Tree

dt_model = Pipeline([("preprocessor", preprocessor), 

                     ("model", DecisionTreeClassifier(class_weight="balanced"))])



# Random Forest

rf_model = Pipeline([("preprocessor", preprocessor), 

                     ("model", RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1))])



# XGBoost

xgb_model = Pipeline([("preprocessor", preprocessor), 

                      # Add a scale_pos_weight to make it balanced

                      ("model", XGBClassifier(scale_pos_weight=(1 - y.mean()), n_jobs=-1))])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.3, random_state=42)
gs = GridSearchCV(lr_model, {"model__C": [1, 1.3, 1.5]}, n_jobs=-1, cv=5, scoring="accuracy")

gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
lr_model.set_params(**gs.best_params_)
lr_model.get_params("model")
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
import eli5

eli5.show_weights(lr_model.named_steps["model"])
preprocessor = lr_model.named_steps["preprocessor"]
ohe_categories = preprocessor.named_transformers_["categorical"].categories_
new_ohe_features = [f"{col}__{val}" for col, vals in zip(cat_features, ohe_categories) for val in vals]
all_features = num_features + new_ohe_features
pd.DataFrame(lr_model.named_steps["preprocessor"].transform(X_train), columns=all_features).head()
eli5.show_weights(lr_model.named_steps["model"], feature_names=all_features)
i = 4

X_test.iloc[[i]]
y_test.iloc[i]
eli5.show_prediction(lr_model.named_steps["model"], 

                     lr_model.named_steps["preprocessor"].transform(X_test)[i],

                     feature_names=all_features, show_feature_values=True)
gs = GridSearchCV(dt_model, {"model__max_depth": [3, 5, 7], 

                             "model__min_samples_split": [2, 5]}, 

                  n_jobs=-1, cv=5, scoring="accuracy")



gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
dt_model.set_params(**gs.best_params_)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
eli5.show_weights(dt_model.named_steps["model"], feature_names=all_features)
eli5.show_prediction(dt_model.named_steps["model"], 

                     dt_model.named_steps["preprocessor"].transform(X_test)[i],

                     feature_names=all_features, show_feature_values=True)
gs = GridSearchCV(rf_model, {"model__max_depth": [10, 15], 

                             "model__min_samples_split": [5, 10]}, 

                  n_jobs=-1, cv=5, scoring="accuracy")



gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)
rf_model.set_params(**gs.best_params_)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
eli5.show_weights(rf_model.named_steps["model"], 

                  feature_names=all_features)
gs = GridSearchCV(xgb_model, {"model__max_depth": [5, 10],

                              "model__min_child_weight": [5, 10],

                              "model__n_estimators": [25]},

                  n_jobs=-1, cv=5, scoring="accuracy")



gs.fit(X_train, y_train)
print(gs.best_params_)

print(gs.best_score_)

xgb_model.set_params(**gs.best_params_)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
from lime.lime_tabular import LimeTabularExplainer
categorical_names = {}

for col in cat_features:

    categorical_names[X_train.columns.get_loc(col)] = [new_col.split("__")[1] 

                                                       for new_col in new_ohe_features 

                                                       if new_col.split("__")[0] == col]
categorical_names
def convert_to_lime_format(X, categorical_names, col_names=None, invert=False):

    """Converts data with categorical values as string into the right format 

    for LIME, with categorical values as integers labels.



    It takes categorical_names, the same dictionary that has to be passed

    to LIME to ensure consistency. 



    col_names and invert allow to rebuild the original dataFrame from

    a numpy array in LIME format to be passed to a Pipeline or sklearn

    OneHotEncoder

    """



    # If the data isn't a dataframe, we need to be able to build it

    if not isinstance(X, pd.DataFrame):

        X_lime = pd.DataFrame(X, columns=col_names)

    else:

        X_lime = X.copy()



    for k, v in categorical_names.items():

        if not invert:

            label_map = {

                str_label: int_label for int_label, str_label in enumerate(v)

            }

        else:

            label_map = {

                int_label: str_label for int_label, str_label in enumerate(v)

            }



        X_lime.iloc[:, k] = X_lime.iloc[:, k].map(label_map)



    return X_lime

convert_to_lime_format(X_train, categorical_names).head()
explainer = LimeTabularExplainer(convert_to_lime_format(X_train, categorical_names).values,

                                 mode="classification",

                                 feature_names=X_train.columns.tolist(),

                                 categorical_names=categorical_names,

                                 categorical_features=categorical_names.keys(),

                                 discretize_continuous=True,

                                 random_state=42)
i = 2

X_observation = X_test.iloc[[i], :]

X_observation
print(f"""\

* True label: {y_test.iloc[i]}

* LR: {lr_model.predict_proba(X_observation)[0]}

* DT: {dt_model.predict_proba(X_observation)[0]}

* RF: {rf_model.predict_proba(X_observation)[0]}

* XGB: {xgb_model.predict_proba(X_observation)[0]}""")
observation = convert_to_lime_format(X_test.iloc[[i], :],categorical_names).values[0]

observation
# Let write a custom predict_proba functions for our models:

from functools import partial



def custom_predict_proba(X, model):

    X_str = convert_to_lime_format(X, categorical_names, col_names=X_train.columns, invert=True)

    return model.predict_proba(X_str)
lr_predict_proba = partial(custom_predict_proba, model=lr_model)

dt_predict_proba = partial(custom_predict_proba, model=dt_model)

rf_predict_proba = partial(custom_predict_proba, model=rf_model)

xgb_predict_proba = partial(custom_predict_proba, model=xgb_model)
explanation = explainer.explain_instance(observation, lr_predict_proba, num_features=5)
explanation.show_in_notebook(show_table=True, show_all=False)
explanation.save_to_file("explanation.html")
print(explanation.local_exp)

print(explanation.intercept)

print(explanation.score)
# dt_predict_proba
explanation = explainer.explain_instance(observation, dt_predict_proba, num_features=5)

explanation.show_in_notebook(show_table=True, show_all=False)

print(explanation.score)
explanation = explainer.explain_instance(observation, rf_predict_proba, num_features=5)

explanation.show_in_notebook(show_table=True, show_all=False)

print(explanation.score)
explanation = explainer.explain_instance(observation, xgb_predict_proba, num_features=5)

explanation.show_in_notebook(show_table=True, show_all=False)

print(explanation.score)
import shap

# Need to load JS vis in the notebook

shap.initjs() 
explainer = shap.TreeExplainer(xgb_model.named_steps["model"])
observations = xgb_model.named_steps["preprocessor"].transform(X_train.sample(1000, random_state=42))

shap_values = explainer.shap_values(observations)
i = 0

shap.force_plot(explainer.expected_value, shap_values[i], 

                features=observations[i], feature_names=all_features)
shap.force_plot(explainer.expected_value, shap_values,

                features=observations, feature_names=all_features)
shap.summary_plot(shap_values, features=observations, feature_names=all_features)
shap.dependence_plot("nr.employed", shap_values, 

                     pd.DataFrame(observations, columns=all_features))
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

from keras.preprocessing.image import load_img, img_to_array
model = InceptionV3()
image_raw = load_img("data/toucan.jpg", target_size=(229, 229))

image_raw
# Convert to numpy array, reshape and preprocess

image = img_to_array(image_raw)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)
predictions = model.predict(image)
decode_predictions(predictions)
model.predict(image).argsort()[0, -5:][::-1]
from lime.lime_image import LimeImageExplainer
explainer = LimeImageExplainer()
explanation = explainer.explain_instance(image[0], model.predict, 

                                         top_labels=2, num_samples=100,

                                         random_seed=42)
from skimage.segmentation import mark_boundaries

from matplotlib import pyplot as plt
temp, mask = explanation.get_image_and_mask(96, positive_only=True, num_features=5, hide_rest=True)

# plot image and mask together

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
temp, mask = explanation.get_image_and_mask(96, positive_only=True, num_features=1, hide_rest=True)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
temp, mask = explanation.get_image_and_mask(779, positive_only=False, num_features=8, hide_rest=True)

plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))