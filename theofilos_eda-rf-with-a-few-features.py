import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import textblob
%pylab inline
df=pd.read_csv('/kaggle/input/tryinclass/train.csv')
df['resource_category'].value_counts().plot.bar();
df['project_grade_level'].value_counts().plot.bar();
df.boxplot(column='cost');
df[df['cost']<3000]['cost'].hist(bins=100);
%pylab inline
plt.rcParams["figure.figsize"] = [15, 5];
df.groupby('posted_date').sum()['cost'].plot();
df['free_lunch_percentage'].hist();
df['project_category'].value_counts().head(20).plot.bar();
df['project_subcategory'].value_counts().head(20).plot.bar();
df['metro_type'].value_counts().plot.bar();
df.iloc[1]
df['project_essay-sentiment'] = df['project_essay'].apply(lambda se: textblob.TextBlob(se).sentiment.polarity)
df['project_title-sentiment'] = df['project_title'].apply(lambda se: textblob.TextBlob(se).sentiment.polarity)
categorical=['project_type','project_category','project_subcategory',
             'project_grade_level','metro_type','resource_category',
             'school_id','state','city','county','district'
            ]
X=df[categorical]
y=df['funded_or_not']
enc = preprocessing.OrdinalEncoder()
enc.fit(X.dropna())
X=enc.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
numerical=['cost','free_lunch_percentage','teachers_project_no',
           'project_title-sentiment','project_essay-sentiment'
          ]
X=df[numerical]
y=df['funded_or_not']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))