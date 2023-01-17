!pip install pydotplus
import pandas as pd
path = "../input/craigslist-carstrucks-data/vehicles.csv"

df = pd.read_csv(path)
df.head()
df.shape
df.dtypes
df.describe()
from pandas_profiling import ProfileReport

profile = ProfileReport(df, minimal=True, title='Used Cars Profiling Report', html={'style':{'full_width':True}})

profile
columns_to_skip = ['id','url','region','region_url','title_status','vin','image_url','description', 'county', 'state', 'lat', 'long']

df = df.drop(columns=columns_to_skip)

df.dtypes
df.shape
df.isna().sum().plot.bar()
columns_to_skip_because_of_null_quantity = ['size']
df = df.drop(columns=columns_to_skip_because_of_null_quantity)
df.shape
df = df.dropna()
df.shape
df.head()
df.nunique()
df = df.drop(columns='model')
df['odometer'].hist()
df.sort_values('odometer', ascending=False).head(500)
df[df.odometer < 300000]['odometer'].hist()
categories = [

    ('0_light', 60000),

    ('1_medium', 120000),

    ('2_heavy', 9999999999)

]



def odocategories(distance):

    for name,value in categories:

        if distance < value:

            return name

    return categories[-1][0]

df['distance'] = df['odometer'].apply(odocategories)

df.head()
df['price'].hist()
df.sort_values('price', ascending=False).head(100)
df[df.price < 100000]['price'].hist(bins=100)
df.sort_values('price').head()
df = df[df.price > 0]
expensiveness_trigger = 20000



df['expensive'] = df.price.map(lambda price: 0 if (price < expensiveness_trigger) else 1)
df.head()
df[df.price > expensiveness_trigger].head()
df.expensive.value_counts()
def cylinder_text_to_number(txt):

    first_letter = txt[0]

    if first_letter.isdigit():

        return int(first_letter)

    else:

        return None

df['cylinder_number'] = df.cylinders.apply(cylinder_text_to_number)

# df.isna()['cylinder_number'].sum()

df = df.dropna()

df.head()
df.condition.value_counts()
conditions = {

    'salvage' : 0,

    'fair': 1,

    'good': 2,

    'excellent': 3,

    'like new': 4,

    'new': 5

}

df['cat_condition'] = df.condition.apply(lambda v:conditions[v])
from sklearn.preprocessing import LabelEncoder

le_manufacturer = LabelEncoder()

df['cat_manufacturer'] = le_manufacturer.fit_transform(df['manufacturer'])

le_fuel = LabelEncoder()

df['cat_fuel'] = le_fuel.fit_transform(df['fuel'])

le_transmission = LabelEncoder()

df['cat_transmission'] = le_transmission.fit_transform(df['transmission'])

le_type = LabelEncoder()

df['cat_type'] = le_type.fit_transform(df['type'])

le_paint_color = LabelEncoder()

df['cat_paint_color'] = le_paint_color.fit_transform(df['paint_color'])

le_distance = LabelEncoder()

df['cat_distance'] = le_distance.fit_transform(df['distance'])

df.head()
df.dtypes
from sklearn.model_selection import train_test_split



y = df['expensive']

X = df.select_dtypes('number').drop(columns=['price', 'expensive', 'odometer'])



X
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



feature_names = list(X.columns)

feature_names
X_train.shape
y_test.shape
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=5, random_state=0)
dtc.fit(X_train, y_train)
dtc.score(X_test, y_test)
from sklearn.metrics import classification_report

y_true = y_test

y_pred = dtc.predict(X_test)

print(classification_report(y_true, y_pred))
dtc.classes_
labelencoder_classes = lambda le: list(zip(le.classes_, range(len(le.classes_))))

print("type")

print(labelencoder_classes(le_type))

print("fuel")

print(labelencoder_classes(le_fuel))
import sklearn.tree as tree

import pydotplus

from sklearn.externals.six import StringIO 

from IPython.display import Image

dot_data = StringIO()

tree.export_graphviz(dtc, 

 out_file=dot_data, 

 class_names=['cheap','expensive'],

 feature_names=feature_names,

 filled=True, # Whether to fill in the boxes with colours.

 rounded=True, # Whether to round the corners of the boxes.

 special_characters=True,

                     proportion=True

                    )

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

Image(graph.create_png())
graph.write_png("used_cars.png")
from sklearn.feature_selection import chi2
y.value_counts()
chi2_results = chi2(X, y)

df_chi2 = pd.DataFrame(chi2_results)

df_chi2.columns = X.columns

df_chi2.index = ['cheap', 'expensive']

df_chi2 = df_chi2.T

df_chi2.cheap.plot.bar()
import cufflinks as cf

cf.go_offline()
df = df[df.year > 1980]
df.groupby('year').count()['price'].iplot()
df_count_yearly = pd.DataFrame(df.groupby(['year', 'type']).count()['price'])

df_count_yearly.columns = ['count']

df_count_yearly.head()
df_pivot = pd.pivot_table(df_count_yearly, columns='type', index='year')['count']

df_pivot.head()
df_pivot.iplot()