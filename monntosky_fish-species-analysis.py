import pandas as pd

df = pd.read_csv("/kaggle/input/fish-market/Fish.csv")

df
import seaborn as sns

sns.heatmap(df.corr(),annot=True)
import plotly.express as px

fig = px.histogram(df, x="Height", y="Weight", color="Species")

fig.show()
X= df.drop('Species',axis=1)

y = (df['Species'])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import *

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
model.score(X_test,y_test)
df_new = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df_new