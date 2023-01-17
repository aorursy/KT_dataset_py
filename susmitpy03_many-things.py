import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

def heatmap(x, y, **kwargs):

    if 'color' in kwargs:

        color = kwargs['color']

    else:

        color = [1]*len(x)



    if 'palette' in kwargs:

        palette = kwargs['palette']

        n_colors = len(palette)

    else:

        n_colors = 256 # Use 256 colors for the diverging color palette

        palette = sns.color_palette("Blues", n_colors) 



    if 'color_range' in kwargs:

        color_min, color_max = kwargs['color_range']

    else:

        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation



    def value_to_color(val):

        if color_min == color_max:

            return palette[-1]

        else:

            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            ind = int(val_position * (n_colors - 1)) # target index in the color palette

            return palette[ind]



    if 'size' in kwargs:

        size = kwargs['size']

    else:

        size = [1]*len(x)



    if 'size_range' in kwargs:

        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]

    else:

        size_min, size_max = min(size), max(size)



    size_scale = kwargs.get('size_scale', 500)



    def value_to_size(val):

        if size_min == size_max:

            return 1 * size_scale

        else:

            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range

            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1

            return val_position * size_scale

    if 'x_order' in kwargs: 

        x_names = [t for t in kwargs['x_order']]

    else:

        x_names = [t for t in sorted(set([v for v in x]))]

    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}



    if 'y_order' in kwargs: 

        y_names = [t for t in kwargs['y_order']]

    else:

        y_names = [t for t in sorted(set([v for v in y]))]

    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}



    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid

    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot



    marker = kwargs.get('marker', 's')



    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [

         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'

    ]}



    ax.scatter(

        x=[x_to_num[v] for v in x],

        y=[y_to_num[v] for v in y],

        marker=marker,

        s=[value_to_size(v) for v in size], 

        c=[value_to_color(v) for v in color],

        **kwargs_pass_on

    )

    ax.set_xticks([v for k,v in x_to_num.items()])

    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')

    ax.set_yticks([v for k,v in y_to_num.items()])

    ax.set_yticklabels([k for k in y_to_num])



    ax.grid(False, 'major')

    ax.grid(True, 'minor')

    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)

    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)



    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])

    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    ax.set_facecolor('#F1F1F1')



    # Add color legend on the right side of the plot

    if color_min < color_max:

        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot



        col_x = [0]*len(palette) # Fixed x coordinate for the bars

        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars



        bar_height = bar_y[1] - bar_y[0]

        ax.barh(

            y=bar_y,

            width=[5]*len(palette), # Make bars 5 units wide

            left=col_x, # Make bars start at 0

            height=bar_height,

            color=palette,

            linewidth=0

        )

        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle

        ax.grid(False) # Hide grid

        ax.set_facecolor('white') # Make background white

        ax.set_xticks([]) # Remove horizontal ticks

        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max

        ax.yaxis.tick_right() # Show vertical ticks on the right 



def corrplot(data, size_scale=500, marker='s'):

    corr = pd.melt(data.reset_index(), id_vars='index')

    corr.columns = ['x', 'y', 'value']

    heatmap(

        corr['x'], corr['y'],

        color=corr['value'], color_range=[-1, 1],

        palette=sns.diverging_palette(20, 220, n=256),

        size=corr['value'].abs(), size_range=[0,1],

        marker=marker,

        x_order=data.columns,

        y_order=data.columns[::-1],

        size_scale=size_scale

    )



    

def sorted_bar_plot(groupby_object,x,y):

    res = groupby_object.aggregate(np.mean).reset_index().sort_values(y)

    pl = sns.barplot(x=x,y=y,data=res,order=res[x])

    for item in pl.get_xticklabels():

        item.set_rotation(90)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.preprocessing import MinMaxScaler





import sklearn



import os

os.chdir("../input/titanic/")
sklearn.__version__
df_train = pd.read_csv("./train.csv")

df_test = pd.read_csv("./test.csv")



# In case of something happening, to avoid running all cells from beginning.

df_train_raw = df_train.copy()

df_test_raw = df_test.copy()
df_train.dtypes
df_train.describe()
df_train.head()
df_train.isnull().sum()
corrplot(df_train.corr())
df_train.Survived.value_counts(normalize=True)
df_train.PassengerId.head().append(df_train.PassengerId.tail())
sns.boxplot(df_train.Survived,df_train.PassengerId)
df_train.drop("PassengerId",axis=1,inplace=True)

df_test.drop("PassengerId",axis=1,inplace=True)
df_train.Pclass.head().append(df_train.Pclass.tail())
df_train.Pclass.value_counts()
sns.boxplot(df_train.Survived,df_train.Pclass,df_train.Sex)
c = df_train.groupby(["Pclass","Survived"])["Pclass"].sum()

c / c.groupby(level=[0]).transform("sum")  * 100
sns.barplot(df_train.Pclass,df_train.Fare)
df_train.corr().Survived
df_train.drop("Fare",axis=1,inplace=True)

df_test.drop("Fare",axis=1,inplace=True)
df_train.Name.head().append(df_train.Name.tail())
df_train.Name.value_counts()[:10]
df_train["Title"] = df_train.Name.str.extract(r"([A-Za-z]+[.])")

df_test["Title"] = df_test.Name.str.extract(r"([A-Za-z]+[.])")
df_train.Title.value_counts()
group = df_train.groupby("Title")["Survived"]

sorted_bar_plot(group,"Title","Survived")
title_low = ["Capt","Don","Rev","Mr"]

title_medium = ["Dr","Col","Major","Master"]

title_high = ["Miss","Mrs","Mile","Mme","Countess","Ms","Lady","Sir"]



title_group = {}

for tl in title_low:

    title_group[tl+"."] = "Low"

for tm in title_medium:

    title_group[tm+"."] = "Medium"

for th in title_high:

    title_group[th+"."] = "High"

    

    

df_train["Title_Group_Impact"] = df_train.Title.map(title_group)

df_test["Title_Group_Impact"] = df_test.Title.map(title_group)
df_train.Title_Group_Impact.head()
group = df_train.groupby("Title_Group_Impact")["Survived"]

sorted_bar_plot(group,"Title_Group_Impact","Survived")
sns.boxplot(df_train.Title_Group_Impact,df_train.Survived)
sns.barplot(df_train.Survived,df_train.Sex)
df_train.Age.isnull().sum() / len(df_train.Age) * 100
median_age = df_train.groupby("Title").Age.aggregate(np.median).sort_values(ascending=False).to_dict()
nan_ages_train = list(df_train.loc[df_train.Age.isnull()].index)

nan_ages_test = list(df_test.loc[df_test.Age.isnull()].index)
for i in nan_ages_train:

    df_train.ix[i,"Age"] = median_age[df_train.ix[i,"Title"]]



for i in nan_ages_test:

    df_test.ix[i,"Age"] = median_age[df_test.ix[i,"Title"]]
sns.distplot(df_train.Age)
sns.boxplot(df_train.Survived,df_train.Age)
sns.barplot(df_train.SibSp,df_train.Survived)
sns.barplot(df_train.Parch,df_train.Survived)
df_train["Family_Size"] = df_train.SibSp + df_train.Parch

df_test["Family_Size"] = df_test.SibSp + df_test.Parch
group = df_train.groupby("Family_Size")["Survived"]

sorted_bar_plot(group,"Family_Size","Survived")
def family_size_impact(x):

    if x in [1,2,3]:

        return "High"

    return "Low"



df_train["Family_Size_Impact"] = df_train.Family_Size.map(family_size_impact)

df_test["Family_Size_Impact"] = df_test.Family_Size.map(family_size_impact)

sns.barplot(df_train.Survived,df_train.Family_Size_Impact)
df_train.drop("Ticket",axis=1,inplace=True)

df_test.drop("Ticket",axis=1,inplace=True)
df_train.drop("Cabin",axis=1,inplace=True)

df_test.drop("Cabin",axis=1,inplace=True)
df_train.head(n=2)
df_train.dtypes
df_train.drop(["Name","SibSp","Parch","Embarked","Title","Family_Size"],axis=1,inplace=True)

df_test.drop(["Name","SibSp","Parch","Embarked","Title","Family_Size"],axis=1,inplace=True)
title_group_impact_ordinal_value = {"High":3,"Medium":2,"Low":1}

family_size_impact_ordinal_value = {"High":2,"Low":1}



df_train.Family_Size_Impact = df_train.Family_Size_Impact.map(family_size_impact_ordinal_value)

df_test.Family_Size_Impact = df_test.Family_Size_Impact.map(family_size_impact_ordinal_value)



df_train.Title_Group_Impact = df_train.Title_Group_Impact.map(title_group_impact_ordinal_value)

df_test.Title_Group_Impact = df_test.Title_Group_Impact.map(title_group_impact_ordinal_value)

min_max_scaler = MinMaxScaler()

df_train.Age = min_max_scaler.fit_transform(np.array(df_train.Age).reshape(-1,1))

df_test.Age = min_max_scaler.fit_transform(np.array(df_test.Age).reshape(-1,1))
print(df_train.Age.min(),df_train.Age.max())

print(df_test.Age.min(),df_test.Age.max())
df_train.Survived = pd.Categorical(df_train.Survived)
df_train.dtypes
df_train.Title_Group_Impact = pd.Series(np.array(df_train.Title_Group_Impact,dtype="int64"))

df_test.Title_Group_Impact = pd.Series(np.array(df_test.Title_Group_Impact,dtype="int64"))
df_train.dtypes
df_train.head()
df_test.head()
def male(x):

    if x == "male":

        return 1

    return 0



df_train["male"] = df_train.Sex.map(male)

df_test["male"] = df_test.Sex.map(male)



df_train.drop("Sex",axis=1,inplace=True)

df_test.drop("Sex",axis=1,inplace=True)
df_train.male = pd.Categorical(df_train.male)

df_test.male = pd.Categorical(df_test.male)
df_train.head()
df_train.dtypes
df_test.dtypes
from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC, SVC



from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier



from keras.models import Sequential

from keras.layers import Dense



from sklearn.metrics import accuracy_score



X = df_train.drop("Survived",axis=1)

y = df_train.Survived
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3, shuffle=True)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
log_reg = LogisticRegression(solver="lbfgs")

log_reg.fit(X_train,y_train)

y_pred = log_reg.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(log_reg,X_train,y_train,cv=20).mean()
gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(gnb,X_train,y_train,cv=20).mean()
svc = SVC(gamma="auto")

svc.fit(X_train,y_train)

y_pred = svc.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(svc,X_train,y_train,cv=20).mean()
l_svc = LinearSVC(max_iter=5000)

l_svc.fit(X_train,y_train)

y_pred = l_svc.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)

cross_val_score(l_svc,X_train,y_train,cv=20).mean()
etc = ExtraTreesClassifier()

etc.fit(X_train,y_train)

y_pred = etc.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(etc,X_train,y_train,cv=20).mean()
rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred = rf.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(rf,X_train,y_train,cv=20).mean()
ab = AdaBoostClassifier()

ab.fit(X_train,y_train)

y_pred = ab.predict(X_val)

print("Accuracy Score: ", accuracy_score(y_val,y_pred)*100)
cross_val_score(ab,X_train,y_train,cv=20).mean()
model = Sequential()

model.add(Dense(16,activation="relu",input_shape=(5,)))

model.add(Dense(32,activation="relu"))

model.add(Dense(64,activation="relu"))

model.add(Dense(8,activation="tanh"))

model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
fitted = model.fit(X_train,y_train,epochs=9,validation_data=(X_val,y_val))
model = Sequential()

model.add(Dense(16,activation="relu",input_shape=(5,)))

model.add(Dense(32,activation="relu"))

model.add(Dense(64,activation="relu"))

model.add(Dense(8,activation="tanh"))

model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X,y,epochs=9)
y_pred = model.predict(df_test)
final_ans = []



for i in y_pred:

    if i >= 0.5:

        final_ans.append(1)

    else:

        final_ans.append(0)
test = pd.read_csv("./test.csv")
ans = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":final_ans})

ans.set_index("PassengerId",inplace=True)
ans.head()
#ans.to_csv("keras_submission.csv")
svc = SVC(gamma="auto")

svc.fit(X,y)

y_pred = etc.predict(df_test)

final_ans = []



for i in y_pred:

    if i >= 0.5:

        final_ans.append(1)

    else:

        final_ans.append(0)

ans = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":final_ans})

ans.set_index("PassengerId",inplace=True)
#ans.to_csv("ETC_submission.csv")