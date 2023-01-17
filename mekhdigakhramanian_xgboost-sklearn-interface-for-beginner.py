import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
df = pd.read_csv('../input/mlcourse/telecom_churn.csv')
df.head()
state_enc = LabelEncoder()
df['State'] = state_enc.fit_transform(df['State'])
df['International plan'] = (df['International plan'] == 'Yes').astype('int')
df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype('int')
df['Churn'] = (df['Churn']).astype('int')
X_train, X_test, y_train, y_test = train_test_split(df.drop('Churn', axis=1), df['Churn'],test_size=0.3, stratify=df['Churn'], random_state=17)
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 1.0,
    'n_estimators': 50
}
xgb_model = XGBClassifier(**params).fit(X_train, y_train)
preds_prob = xgb_model.predict(X_test)
predicted_labels = preds_prob > 0.5
print("Accuracy and F1 on the test set are: {:.2} and {:.2}".format(
    round(accuracy_score(y_test, predicted_labels), 3),
    round(f1_score(y_test, predicted_labels), 3)))