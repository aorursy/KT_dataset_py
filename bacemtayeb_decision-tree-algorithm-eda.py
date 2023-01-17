import pandas as pd #Data loading + processing
import numpy as np #Linear Algebra 
import matplotlib.pyplot as plt #Visualization
from sklearn import tree #Classifier
from sklearn.model_selection import train_test_split #Cross Validation
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
plt.rcParams["figure.figsize"] = [16, 12]
data = pd.read_csv('../input/dream_market_cocaine_listings.csv')
df = pd.DataFrame(data)
df.shape
#Some useful insights
pd.Series([df.shape[0],df.shape[1],df.isnull().sum().max(),df.ships_from.value_counts().index[0],df.ships_to.value_counts().index[0],df.groupby('ships_from').grams.value_counts().index[0],df.product_title.max(),df.vendor_name.value_counts().index[0],df.ships_from_to.value_counts().index[0]],index = ['Number Of Instances','Number Of Columns','NaN Values','Ship From','Ship To','Ship From based on grams','Most Sold Product',"Vendor",'Market'])
#Even more better insights
df.describe()
#Whoa we have a lot of columns let's just drop usless ones
df.columns
new_df = df.drop(columns={'ships_to_US', 'ships_from_US',
       'ships_to_NL', 'ships_from_NL', 'ships_to_FR', 'ships_from_FR',
       'ships_to_GB', 'ships_from_GB', 'ships_to_CA', 'ships_from_CA',
       'ships_to_DE', 'ships_from_DE', 'ships_to_AU', 'ships_from_AU',
       'ships_to_EU', 'ships_from_EU', 'ships_to_ES', 'ships_from_ES',
       'ships_to_N. America', 'ships_from_N. America', 'ships_to_BE',
       'ships_from_BE', 'ships_to_WW', 'ships_from_WW', 'ships_to_SI',
       'ships_from_SI', 'ships_to_IT', 'ships_from_IT', 'ships_to_DK',
       'ships_from_DK', 'ships_to_S. America', 'ships_from_S. America',
       'ships_to_CH', 'ships_from_CH', 'ships_to_BR', 'ships_from_BR',
       'ships_to_CZ', 'ships_from_CZ', 'ships_to_SE', 'ships_from_SE',
       'ships_to_CO', 'ships_from_CO', 'ships_to_CN', 'ships_from_CN',
       'ships_to_PL', 'ships_from_PL', 'ships_to_GR', 'ships_from_GR','product_link','vendor_link'})
sns.pairplot(new_df)
sns.distplot(new_df["cost_per_gram"],bins=20, hist=True, rug=True)
print('mean %r vs median %r.'%(new_df.mean().sum(),new_df.median().sum()))
print('As %r is greater than %r, our data is right skewed.'%(new_df.mean().sum(),new_df.median().sum()))
new_df.skew()
new_df.hist(alpha=0.5, figsize=(16, 10))
df.plot.scatter(y = 'cost_per_gram_pure', x = 'quality')
new_df.mean()
df.plot.scatter(y = 'cost_per_gram_pure', x = 'cost_per_gram')
features = new_df[['btc_price','grams','rating']]
new_df['quality_rating'] = np.where(new_df['quality']>=85, 'high', 'medium')
label = new_df['quality_rating']
clf = tree.DecisionTreeClassifier()
clf.fit(features,label)
clf.predict([[1.2,20,20]])
#Ensemble Voting Classifiers
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
eclf1 = VotingClassifier(estimators=[('dts', clf), ('lr', clf1), ('rnf', clf2)], voting='soft')
probas = [c.fit(features, label).predict([[1.2,20,20]]) for c in (clf, clf1, clf2,eclf1)]