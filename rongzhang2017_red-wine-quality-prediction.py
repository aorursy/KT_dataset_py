#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected =True)
import plotly.graph_objs as go
from plotly import tools
from wordcloud import WordCloud    
import matplotlib.pyplot as plt 

#Loading data
wine = pd.read_csv('../input/winequality-red.csv')
wine.head()
#change column names
wine.rename(columns={"fixed acidity":"fixed_acidity", "residual sugar": "residual_sugar"}, inplace=True)
#Information about the data columns
wine.info()
#Classify quality into three groups
def quality_group(x):   
    if x <= 4:
        qg = "Low"
    elif (x > 4) & (x <= 6):
        qg = "Medium"
    else:
        qg = "High"
    return qg  
wine['qualitygroup'] = wine.quality.apply(quality_group)
wine.head()
wine_df = wine.iloc[:1599,:]
donut= wine.qualitygroup.value_counts()
labels = wine.qualitygroup.value_counts().index

#Creat figure
fig = {
    "data":
    [
        {
            "values": donut,
            "labels": labels,
            "domain": {"x": [0, 1]},
            "name": "Quality Rate",
            "hoverinfo": "label+percent+name",
            "hole": .4,
            "type": "pie"
        }, 
    ],
    "layout":
    {
        "title":"Composition of wine quality",
        "annotations":
        [
            { 
                "font":{"size":20},
                "showarrow":False,
                "text": "",
                "x": 0,
                "y": 1
            },
        ]
    }
}
iplot(fig)
# 100 wines 'fixed acidity', 'residual sugar' and 'alcohol' scores comparison with Scatter 3D Plot
wine_small_df = wine.iloc[:100,:]
trace = go.Scatter3d(
    x=wine_small_df.fixed_acidity,
    y=wine_small_df.residual_sugar,
    z=wine_small_df.alcohol,
    text= wine_small_df.qualitygroup,
    mode='markers',
    marker=dict(
        size=12,
        #color= z,          #set color to an array/list of desired value (plotly.ly)
#When we enters 'Fork Notebook' he describes 'z'. But why doesn't he recognize this right now? 
        colorscale='Viridis',   #Choose a colorscale
        opacity=0.8
    )
)
data = [trace]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv')
bins = (0, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])
sns.countplot(wine['quality'])
#seperate respons variables and independant variable
X = wine.drop('quality', axis = 1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#Let's see how our model performed
print(classification_report(y_test, pred_rfc))