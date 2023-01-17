import re
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")
data_frame = pd.read_csv("/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv")
data_frame.head()
data_frame.info()
data_frame["Rating"].value_counts()
sns.countplot(x="Rating", data=data_frame)
plt.show()
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
# Remove special characters from the sentence
def clean_text(sentence):
    
    # Convert to lower case
    sentence = sentence.lower()
    # split the sentence
    sentence = sentence.split()
    # Join the sentence
    sentence = " ".join(sentence)
    # Remove special characters from the sentence
    sentence = re.sub(f'[{re.escape(string.punctuation)}]', "", sentence)
    
    return sentence
data_frame["Review"] = data_frame["Review"].apply(clean_text)
x_train, x_test, y_train, y_test = train_test_split(data_frame["Review"], data_frame["Rating"],test_size=0.2, random_state=42)
# Apply Tfidf Vectorizer to convert sentence to tokens
tfidf = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
tfidf.fit(data_frame["Review"])
x_train_vector = tfidf.transform(x_train)
x_test_vector = tfidf.transform(x_test)
# Classes are imbalanced
# SMOTE to over sample and balance classes.
x_smote, y_smote = SMOTE().fit_resample(x_train_vector, y_train)
def evaluation_metric(y_test, y_hat, model_name):
    
    accuracy = accuracy_score(y_hat, y_test)
    print("Model: ", model_name)
    print("\nAccuracy: ", accuracy)
    print(classification_report(y_hat, y_test))
    
    plt.figure(figsize=(10,6))
    sns.heatmap(confusion_matrix(y_hat, y_test), annot=True, fmt=".2f")
    plt.show()
    return accuracy
lr_model = LogisticRegression()
lr_model.fit(x_smote, y_smote)
lr_preds = lr_model.predict(x_test_vector)
lr_accuracy = evaluation_metric(lr_preds, y_test, "Logistic Regression")
rf_model = RandomForestClassifier()
rf_model.fit(x_smote, y_smote)
rf_preds = rf_model.predict(x_test_vector)
rf_accuracy = evaluation_metric(rf_preds, y_test, "Random Forest Classifier")
xgb_model = XGBClassifier(max_depth=10,random_state=1,learning_rate=0.05,seed=1)
xgb_model.fit(x_smote, y_smote)
xgb_preds = xgb_model.predict(x_test_vector)
xgb_accuracy = evaluation_metric(xgb_preds, y_test, "XGB Classifier")
lgb_model = LGBMClassifier()
lgb_model.fit(x_smote, y_smote)
lgb_preds = lgb_model.predict(x_test_vector)
lgb_accuracy = evaluation_metric(lgb_preds, y_test, "LGBM Classifier")
x = ["Random Forest", "Logistic Regression", "XGB Classifier", "LGBM Classifier"]
y = [rf_accuracy, lr_accuracy, xgb_accuracy, lgb_accuracy]
plt.bar(x=x, height=y)
plt.title("Algorithm Accuracy Comparison")
plt.xticks(rotation=15)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()
