# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!pip install chart_studio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
import plotly.offline as py
from plotly.offline import plot
import plotly.graph_objs as go
sns.set_style("darkgrid")
plt.style.use('ggplot')
data = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
df = data.copy()
df.info()
#Remove unwanted columns (object columns)
for col in df.select_dtypes('object'):
    df = df.drop([col], axis=1)
    
df.head(3)
df.isnull().sum()
#Drop column with too many null values and non_wanted columns
df = df.drop(["has_urgency_banner"], axis=1)
df = df.drop(['badge_local_product','badge_fast_shipping','shipping_is_express','inventory_total',"product_variation_inventory"], axis=1)
#Replacing nan values by 0
for c in df.columns:
    if df[c].isnull().sum() > 40:
        df[c] = df[c].replace(np.nan, 0)
#Keeping some features
df["product_variation_size_id"] = data["product_variation_size_id"]
df["origin_country"] = data["origin_country"]
df["product_color"] = data["product_color"]

df.head(3)
df['product_variation_size_id'].value_counts().head(60)
def encoding_prod_var(name):
    if name == 28 | 29 \
    or name == "Size -XXS" \
    or name == "SIZE-XXS":
        return "XXS"
    elif name == 30 | 31 \
    or name == "XS." \
    or name == "Size-XS" \
    or name == "SIZE XS":
        return "XS"
    elif name == 32 | 33 \
    or name == "S." \
    or name == "Suit-S" \
    or name == "Size S" \
    or name == "size S" \
    or name == "Size--S" \
    or name == "Size-S" \
    or name == "S Pink" \
    or name == "s":
        return "S"
    elif name == 34 \
    or name == "M." \
    or name == "Size M":
        return "M"
    elif name == 35 \
    or name == "L." \
    or name == "SizeL":
        return "L"
    elif name == 36 \
    or name == "X   L":
        return "XL"
    elif name == 37 \
    or name == "2XL":
        return "XXL"
    elif name == 'XXXS' \
    or name == 'XXS' \
    or name == 'XS' \
    or name == 'S' \
    or name == 'M' \
    or name == 'L' \
    or name == 'XL' \
    or name == 'XXL' \
    or name == 'XXXXL' \
    or name == 'XXXXXL':
        return name
    else:
        return "Other"
    
df['product_variation_size_id'] = df['product_variation_size_id'].replace(np.nan, "Other")
df['product_variation_size_id'] = df['product_variation_size_id'].apply(encoding_prod_var)

df["product_variation_size_id"].value_counts()
#Count by size : Data Analysis
fig = px.bar(df['product_variation_size_id'].value_counts(), orientation="h", color=df['product_variation_size_id'].value_counts().index, color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Count', 
                                'index':'Size',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Count by size"
)

fig.show()
#Changing origin country by Other
df['origin_country'] = df['origin_country'].replace("VE", "Other")
df['origin_country'] = df['origin_country'].replace("SG", "Other")
df['origin_country'] = df['origin_country'].replace("GB", "Other")
df['origin_country'] = df['origin_country'].replace("AT", "Other")
df['origin_country'].value_counts()
#Count by origin_country : Data Analysis
fig = px.bar(df['origin_country'].value_counts(), orientation="v", color=df['origin_country'].value_counts().index, color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Count', 
                                'index':'Origin country',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Count by Origin country"
)

fig.show()
#Encoding product colors

def encoding_prod_color(name):
    if name == "armygreen" \
    or name == "khaki" \
    or name == "camouflage"\
    or name == "mintgreen" \
    or name == "lightgreen" \
    or name == "lightkhaki" \
    or name == "Army green" \
    or name == "army green" \
    or name == "darkgreen" \
    or name == "Green" \
    or name == "fluorescentgreen" \
    or name == "applegreen" \
    or name == "navy":
        return "green"
    
    elif name == "Black" \
    or name == "black & white" \
    or name == "black & blue" \
    or name == "coolblack" \
    or name == "black & green" \
    or name == "black & yellow":
        return "black"
    
    elif name == "navyblue" \
    or name == "lightblue" \
    or name == "skyblue" \
    or name == "Blue" \
    or name == "darkblue" \
    or name == "navy blue" \
    or name == "navyblue & white" \
    or name == "lakeblue":
        return "blue"
    
    elif name == "Yellow" \
    or name == "lightyellow" \
    or name == "star":
        return "yellow"
    
    elif name == "offwhite" \
    or name == "White" \
    or name == "whitefloral" \
    or name == "white & black" \
    or name == "white & green":
        return "white"
    
    elif name == "rosered" \
    or name == "rose" \
    or name == "Pink" \
    or name == "Rose" \
    or name == "pink & grey" \
    or name == "floral" \
    or name == "lightpink" \
    or name == "pink & white" \
    or name == "pink & black" \
    or name == "pink & blue" \
    or name == "dustypink":
        return "pink"
    
    elif name == "Red" \
    or name == "rouge" \
    or name == "lightred" \
    or name == "coralred" \
    or name == "watermelonred" \
    or name == "Rouge":
        return "red"
    
    elif name == "Orange" \
    or name == "orange-red" \
    or name == "apricot":
        return "orange"
    
    elif name == 'coffee':
        return "brown"
    
    elif name == "lightgrey" \
    or name == "gray" \
    or name == "Grey" \
    or name == "grey":
        return "grey"
    
    elif name == 'white' \
    or name == 'black' \
    or name == 'yellow' \
    or name == 'pink' \
    or name == 'red' \
    or name == 'green' \
    or name == 'orange' \
    or name == 'grey' \
    or name == 'brown' \
    or name == "purple" \
    or name == "blue" \
    or name == 'beige':
        return name
    
    else:
        return "other"
    
df['product_color'] = df['product_color'].replace(np.nan, "Other")
df['product_color'] = df['product_color'].apply(encoding_prod_color)
df['product_color'].value_counts().head(50)
#Count by origin_country : Data Analysis
fig = px.bar(df['product_color'].value_counts(), orientation="v", color=df['product_color'].value_counts().index, color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Count', 
                                'index':'Product colors',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Count by Product colors"
)

fig.show()
def encoding_us(item):
    if item == 10 \
    or item == 50 \
    or item == 100 \
    or item == 1000 \
    or item == 20000:
        return item
    elif item == 5000 \
    or item == 10000:
        return 20000
    else:
        return 10
    
df["units_sold"] = df["units_sold"].apply(encoding_us)
df["units_sold"].value_counts()
#Count by units sold
sns.countplot(df["units_sold"])
plt.figure(figsize=(25, 20))
sns.heatmap(df.corr(), annot=True)
#Import some librairies
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score
df = pd.get_dummies(df, columns = ['product_color'],
                        prefix = "Color_",
                        drop_first = True)

df = pd.get_dummies(df, columns = ["product_variation_size_id"],
                   prefix = "Size_",
                   drop_first = True)

df = pd.get_dummies(df, columns = ["origin_country"],
                   prefix = "Origin_",
                   drop_first = True)

df.head()
#Spliting
X = df.drop(["units_sold"], axis=1)
y = df["units_sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
r = 42

DTC = DecisionTreeClassifier(random_state = r)
RFC = RandomForestClassifier(random_state = r)
ADA = AdaBoostClassifier(RandomForestClassifier(random_state = r),
                                       learning_rate = 0.01) 
GBC = GradientBoostingClassifier(random_state = r)
KNN = KNeighborsClassifier(n_neighbors = 10)
XGB = XGBClassifier()

classifiers = [DTC, RFC, ADA, GBC, KNN, XGB]
classifiers_names = ['Decision Tree',
                     'Random Forest',
                     'AdaBoost - Random Forest',
                     'Gradient Boosting',
                     'KNeighborsClassifier',
                     'XG Boost']
acc_mean = []

for cl in classifiers:
    acc = cross_val_score(estimator = cl, X = X_train, y  = y_train, cv = 2)
    acc_mean.append(acc.mean()*100)
    
acc_df = pd.DataFrame({'Classifiers': classifiers_names,
                       'Accuracies Mean': acc_mean})

acc_df.sort_values('Accuracies Mean',ascending=False)
#from sklearn.model_selection import GridSearchCV

#n_estimators = [100, 300, 500, 800, 1200]
#max_depth = [5, 8, 15, 25, 30]
#min_samples_split = [2, 5, 10, 15, 100]
#min_samples_leaf = [1, 2, 5, 10] 

#hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#              min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

#gridF = GridSearchCV(RFC, hyperF, cv = 4, verbose = 1, n_jobs = -1)
#bestF = gridF.fit(X_train, y_train)
#y_pred = gridF.predict(X_test)
#print(classification_report(y_test, y_pred))
#print(gridF.best_params_)
#Final model
final_model = RandomForestClassifier(max_depth = 25, min_samples_leaf = 1,
                                     min_samples_split =  5, n_estimators = 100)

final_model.fit(X_train, y_train)
y_pred_final_model = final_model.predict(X_test)
accuracy_score(y_test, y_pred_final_model)
f1_score(y_test, y_pred_final_model, average='weighted')
recall_score(y_test, y_pred_final_model, average='weighted')