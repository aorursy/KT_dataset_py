import pandas as pd
address = '../input/creditcardfraud/creditcard.csv'

df = pd.read_csv(address)
df.head() # Strange labels
df['Class'].value_counts()
# from imblearn.over_sampling import RandomOverSampler

# oversample = RandomOverSampler(sampling_strategy=0.5)

X = df.drop('Class',1)

y = df['Class']

# X, y = oversample.fit_sample(X, y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
import xgboost as xgb
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X, y)
xgb_model.score(X_train,y_train)
xgb_model.score(X_test,y_test)
y_predict = xgb_model.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))

confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])

confusion
import seaborn as sns
sns.heatmap(confusion, annot=True)
print(classification_report(y_test, y_predict))