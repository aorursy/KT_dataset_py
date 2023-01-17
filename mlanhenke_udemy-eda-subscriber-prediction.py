import datetime

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler

pd.pandas.set_option('display.max_columns',None)
df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
print(f'Shape: {df.shape} \n')

print(df.info())
df.head()
df.describe()
num_features = [f for f in df.columns if df[f].dtype in ['int64','float64'] and not '_id' in f]

cat_features = [f for f in df.columns if df[f].dtype in ['object','bool']]



print(f'Number of numerical features: {len(num_features)}')

print(f'Number of categorical features: {len(cat_features)}')
features_nan = [f for f in df.columns if df[f].isnull().sum() > 1]

print(f'Number of missing features: {len(features_nan)}')
def plot_multiple_dist(rows, cols, features, fx, fy):

    sns.set_style('dark')

    plt.figure(figsize=(fx,fy))



    for idx, feature in enumerate(features):

        plt.subplot(rows, cols, idx + 1)

        g = sns.distplot(df[feature], color='lightblue')

        skewness = np.round(df[feature].skew(),2)

        g.set_xlabel(f'Skewness: {skewness}', fontsize=12, weight='bold')

        g.set_ylabel('', fontsize=12)

        g.set_title(f'Distribution of {feature}', fontsize=14, weight='bold')

    plt.subplots_adjust(wspace=.3, hspace=.5)

    plt.show()
## distribution of numerical values

plot_multiple_dist(3,2,num_features,14,10)
## log transform all numerical variables

for feature in num_features:

    df[feature+'_log'] = np.log(df[feature] + 1)

num_log_features = [f for f in df.columns if '_log' in f]
plot_multiple_dist(3,2,num_log_features,14,10)
## correlation matrix

plt.figure(figsize=(14,9))

g = sns.heatmap(df[num_log_features].corr(), linewidths=.5, annot=True, cmap='Blues', cbar=False)

g.set_title('Correlation of Numerical Variables', fontsize=14, weight='bold')

g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=10)

g.set_yticklabels(g.get_xticklabels(),rotation=45, fontsize=10)

plt.show()
## most prominent subjects

print('Number of Subjects Total:', df['subject'].nunique(), '\n')

print(df['subject'].value_counts())
## plot numerical distribution over all subjects

plt.figure(figsize=(18,24))



plt.subplot(6,1,1)

gc = sns.countplot('subject', data=df, palette='pastel')

gc.set_xlabel('', fontsize=12)

gc.set_ylabel('Count', fontsize=12)

gc.set_title('Distribution by Subject', fontsize=14, weight='bold')



for idx, feature in enumerate(num_log_features):

    if 'price_log' in feature: feature = 'price'

    plt.subplot(6, 1, idx + 2)

    g = sns.boxplot(x='subject', y=feature, data=df, palette='GnBu_d')

    g.set_xlabel('', fontsize=12)

    g.set_ylabel(feature, fontsize=12)

    g.set_title(f'Distribution by {feature}', fontsize=14, weight='bold')



plt.subplots_adjust(hspace=.7, wspace=.4)

plt.show()
## course level overview

print('Number of Skill-Levels:', df['level'].nunique(), '\n')

print(df['level'].value_counts())
## plot numerical distribution over all skill levels

plt.figure(figsize=(18,24))



plt.subplot(6,1,1)

gc = sns.countplot('level', data=df, palette='pastel')

gc.set_xlabel('', fontsize=12)

gc.set_ylabel('Count', fontsize=12)

gc.set_title('Distribution by Level', fontsize=14, weight='bold')



for idx, feature in enumerate(num_log_features):

    if 'price_log' in feature: feature = 'price'

    plt.subplot(6, 1, idx + 2)

    g = sns.boxplot(x='level', y=feature, data=df, palette='GnBu_d')

    g.set_xlabel('', fontsize=12)

    g.set_ylabel(feature, fontsize=12)

    g.set_title(f'Distribution by {feature}', fontsize=14, weight='bold')



plt.subplots_adjust(hspace=.7, wspace=.4)

plt.show()
plt.figure(figsize=(14,6))

g = sns.countplot(df['is_paid'], palette='pastel')

g.set_xlabel('Paid', fontsize=12)

g.set_ylabel('Count', fontsize=12)

g.set_title('Amount of Paid vs. Free Courses', fontsize=14, weight='bold')

plt.show()
## create datetime features

df['published_timestamp'] = pd.to_datetime(df['published_timestamp'])

df['published_year'] = df['published_timestamp'].dt.year

df['published_month'] = df['published_timestamp'].dt.month

df['published_day'] = df['published_timestamp'].dt.day_name()



publish_features = [f for f in df.columns if 'published' in f and not '_timestamp' in f]
## plot against subscriber count

plt.figure(figsize=(18,12))



for idx, feature in enumerate(publish_features):

    if 'price_log' in feature: feature = 'price'

    plt.subplot(3, 1, idx + 1)

    g = sns.boxplot(x=feature, y='num_subscribers_log', data=df, palette='GnBu_d')

    g.set_xlabel('', fontsize=12)

    g.set_ylabel(feature, fontsize=12)

    g.set_title(f'Subscriber Count per {feature}', fontsize=14, weight='bold')



plt.subplots_adjust(hspace=.7, wspace=.4)

plt.show()
## create new feature course age

current_year = datetime.datetime.now().year

df['age'] = current_year - df['published_year']



plt.figure(figsize=(18,3))

g = sns.boxplot(x='age', y='num_subscribers_log', data=df, palette='GnBu_d');
comment_words = ''

stopwords = set(STOPWORDS)



for s in df['course_title']:

    s = str(s)

    tokens = s.split()

    for i in range(len(tokens)):

        tokens[i] = tokens[i].lower()

    

    comment_words += ' '.join(tokens)+' '



wordcloud = WordCloud(width=800,height=800,

                      background_color='white',

                      stopwords=stopwords,

                      min_font_size=10).generate(comment_words)



plt.figure(figsize=(14,8), facecolor=None)

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
## create dummies for categorical features

num_cols = ['price','num_reviews_log','num_lectures_log','content_duration_log','age']

dummies = pd.get_dummies(df[['is_paid','level','subject']],drop_first=True)



X = df[num_cols].merge(dummies, left_index=True, right_index=True)

y = df['num_subscribers']
## Split into Training and Test Data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)



col_names = X_train.columns



## Apply Standard Scaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
## Model building and prediction

rf_model = RandomForestRegressor(n_estimators=500, random_state=0)



rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)



mse = mean_squared_error(y_pred, y_test)

mae = mean_absolute_error(y_pred, y_test)
## evaluation

print(f'MSE: {mse}')

print(f'MAE: {mae}')



imp_features = pd.Series(rf_model.feature_importances_, index=col_names).sort_values(ascending=False)

plt.figure(figsize=(14,6))

g = imp_features.plot.bar();

g.set_xticklabels(g.get_xticklabels(), rotation=45);