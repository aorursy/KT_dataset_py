import pandas as pd

import plotly.express as px

import numpy as np

import datetime

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.figure_factory as ff
src = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv",parse_dates=True)
src.head()
print(f"No of rows: {src.shape[0]}")

print(f"No of columns: {src.shape[1]}")
print(f"Columns in the dataset\n{pd.Series(src.columns).T}")
src.dtypes
src.nunique()
for i in src.columns[1:]:

    print(i)

    print(src[i].unique())
print(f"% of missing values\n{np.round(src.isnull().mean()*100,2)}")
def to_date(x):

    return datetime.datetime.strptime(x,"%m/%d/%y")



src["b_day"] = src["b_day"].map(to_date)

print(f"datatype: {src.b_day.dtype}\nFew Unique Values\n{src.b_day.unique()[:5]}")
src = src.reset_index(drop=True)

src.insert(7,"height_ft",0) # insert a new column "height_ft" and assign a value 0

src.insert(8,"height_cm",0)# insert a new column "height_cm" and assign a value 0

for i,j in enumerate(src["height"]):

    split = j.split(" / ") # splitting the text by " / "

    src.loc[i,"height_ft"] = split[0].strip() #first element of the split is height in ft

    src.loc[i,"height_cm"] = float(split[1].strip())*100 #second element of the split is height in m. Multiplying it with 100 to convert to cnm
src.head()
src = src.reset_index(drop=True)

src.insert(10,"weight_lbs",0)

src.insert(11,"weight_kg",0)

for i,j in enumerate(src["weight"]):

    split = j.split(" / ")

    src.loc[i,"weight_lbs"] = float(split[0].replace("lbs.","").strip())

    src.loc[i,"weight_kg"] = float(split[1].replace("kg.","").strip())
src.head()
"All the Salaries are in $" if all(src["salary"].str.startswith("$")) else "Not All the Salaries are in $"
src["salary"] = src["salary"].str.replace("$","",regex=False).astype("float64")

src.head()
src.drop(columns=["height","height_ft","weight","weight_lbs"],inplace=True)
src.dtypes
src.head()
src["team"].fillna("Not Known",inplace=True)

src["college"].fillna("Not Known",inplace=True)
src.insert(8,"bmi",0)

src.insert(9,"bmi_class",0)

src["bmi"] = np.round(src["weight_kg"] / ((src["height_cm"]/100)**2),1)

src.loc[src["bmi"]<18.5,"bmi_class"] = "underweight"

src.loc[(src["bmi"]>=18.5) & (src["bmi"]<=24.9),"bmi_class"] = "normal"

src.loc[(src["bmi"]>=25) & (src["bmi"]<=29.9),"bmi_class"] = "overweight"

src.loc[src["bmi"]>=30,"bmi_class"] = "obese"
def age_calc(dob):

    today = datetime.datetime.today()

    return np.floor(((today-dob).days)/365)

src.insert(6,"age",0)

src["age"] = src["b_day"].map(age_calc)
src.head()
fig = make_subplots(rows=2, cols=3, subplot_titles=("Age", "Rating", "Salary","BMI","Height","Weight"))



fig.add_trace(

    go.Histogram(x=src["age"]),

    row=1, col=1

)



fig.add_trace(

    go.Histogram(x=src["rating"]),

    row=1, col=2

)



fig.add_trace(

    go.Histogram(x=src["salary"]),

    row=1, col=3

)



fig.add_trace(

    go.Histogram(x=src["bmi"]),

    row=2, col=1

)



fig.add_trace(

    go.Histogram(x=src["height_cm"]),

    row=2, col=2

)



fig.add_trace(

    go.Histogram(x=src["weight_kg"]),

    row=2, col=3

)



fig.update_layout(title_text="Distribution of Numerical Variables",showlegend=False)
pd.set_option('display.float_format', lambda x: '%.1f' % x)

src[["age", "rating", "salary","bmi","height_cm","weight_kg"]].describe()
src.skew()
teams = pd.DataFrame(src["team"].value_counts()).reset_index()

teams.columns=["team","count"]



pos = pd.DataFrame(src["position"].value_counts()).reset_index()

pos.columns=["position","count"]



cntry = pd.DataFrame(src["country"].value_counts()).reset_index()

cntry.loc[cntry["country"]<(0.005*cntry["country"].sum()),"index"] = "Others"

cntry = cntry.groupby("index").sum().reset_index()

cntry.columns=["country","count"]

cntry.sort_values(by="count",ascending=False,inplace=True)



coll = pd.DataFrame(src["college"].value_counts()).reset_index()

coll.loc[coll["college"]<(0.01*coll["college"].sum()),"index"] = "Others"

coll = coll.groupby("index").sum().reset_index()

coll.columns=["college","count"]

coll.sort_values(by="count",ascending=False,inplace=True)



bmi_dist = pd.DataFrame(src["bmi_class"].value_counts()).reset_index()

bmi_dist.columns=["bmi_class","count"]



fig = make_subplots(rows=4, cols=2,specs=[[{},{}],

                                          [{"colspan": 2},None],

                                         [{"colspan": 2},None],

                                         [{"colspan": 2},None]],

                    subplot_titles=("Position", "BMI_Class","Country", "Team","College"))



fig.add_trace(

    go.Bar(y=pos["count"],x=pos["position"]),

    row=1, col=1

)



fig.add_trace(

    go.Bar(y=bmi_dist["count"],x=bmi_dist["bmi_class"]),

    row=1, col=2

)



fig.add_trace(

    go.Bar(y=cntry["count"],x=cntry["country"]),

    row=2, col=1

)



fig.add_trace(

    go.Bar(y=teams["count"],x=teams["team"]),

    row=3, col=1

)



fig.add_trace(

    go.Bar(y=coll["count"],x=coll["college"]),

    row=4, col=1

)



fig.update_layout(title_text="Distribution of Categorical Variables",showlegend=False,height=1400)
px.scatter_matrix(src[["age","bmi","height_cm","weight_kg","rating","salary"]],height=1500)
cols = ["age","bmi","height_cm","weight_kg","rating","salary"]

fig = ff.create_annotated_heatmap(np.round(src[cols].corr().values,2),x=cols,y=cols)

fig.show()
fig = make_subplots(rows=6, cols=1)



fig.add_trace(

    go.Box(x=src["team"],y=src["age"]),

    row=1, col=1

)



fig.add_trace(

    go.Box(x=src["team"],y=src["bmi"]),

    row=2, col=1

)

fig.add_trace(

    go.Box(x=src["team"],y=src["height_cm"]),

    row=3, col=1

)





fig.add_trace(

    go.Box(x=src["team"],y=src["weight_kg"]),

    row=4, col=1

)



fig.add_trace(

    go.Box(x=src["team"],y=src["salary"]),

    row=5, col=1

)



fig.add_trace(

    go.Box(x=src["team"],y=src["rating"]),

    row=6, col=1

)





fig.update_yaxes(title_text="Age", row=1, col=1)

fig.update_yaxes(title_text="BMI", row=2, col=1)

fig.update_yaxes(title_text="Height", row=3, col=1)

fig.update_yaxes(title_text="Weight", row=4, col=1)

fig.update_yaxes(title_text="Salary", row=5, col=1)

fig.update_yaxes(title_text="Rating", row=6, col=1)



fig.update_layout(title_text="Relationship B/w Team and Important Numerical Variables",showlegend=False,height=2000)
fig = make_subplots(rows=6, cols=1)



fig.add_trace(

    go.Box(x=src["position"],y=src["age"]),

    row=1, col=1

)



fig.add_trace(

    go.Box(x=src["position"],y=src["bmi"]),

    row=2, col=1

)

fig.add_trace(

    go.Box(x=src["position"],y=src["height_cm"]),

    row=3, col=1

)





fig.add_trace(

    go.Box(x=src["position"],y=src["weight_kg"]),

    row=4, col=1

)



fig.add_trace(

    go.Box(x=src["position"],y=src["salary"]),

    row=5, col=1

)



fig.add_trace(

    go.Box(x=src["position"],y=src["rating"]),

    row=6, col=1

)





fig.update_yaxes(title_text="Age", row=1, col=1)

fig.update_yaxes(title_text="BMI", row=2, col=1)

fig.update_yaxes(title_text="Height", row=3, col=1)

fig.update_yaxes(title_text="Weight", row=4, col=1)

fig.update_yaxes(title_text="Salary", row=5, col=1)

fig.update_yaxes(title_text="Rating", row=6, col=1)



fig.update_layout(title_text="Relationship B/w Position and Important Numerical Variables",showlegend=False,height=1700)
fig = make_subplots(rows=3, cols=2)



fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["age"]),

    row=1, col=1

)



fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["bmi"]),

    row=1, col=2

)

fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["height_cm"]),

    row=2, col=1

)





fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["weight_kg"]),

    row=2, col=2

)



fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["salary"]),

    row=3, col=1

)



fig.add_trace(

    go.Box(x=src["bmi_class"],y=src["rating"]),

    row=3, col=2

)





fig.update_yaxes(title_text="Age", row=1, col=1)

fig.update_yaxes(title_text="BMI", row=1, col=2)

fig.update_yaxes(title_text="Height", row=2, col=1)

fig.update_yaxes(title_text="Weight", row=2, col=2)

fig.update_yaxes(title_text="Salary", row=3, col=1)

fig.update_yaxes(title_text="Rating", row=3, col=2)



fig.update_layout(title_text="Relationship B/w BMI Class and Important Numerical Variables",showlegend=False,height=1700)
src1 = src[["height_cm","weight_kg","bmi","bmi_class"]].copy()

src1["bmi"] = (src1["bmi"]-src1["bmi"].min()) / (src1["bmi"].max()-src1["bmi"].min())

px.scatter(src1,x="height_cm",y="weight_kg",size="bmi",color="bmi",trendline="lowess",title="Weight, Height, BMI")
pos_bmi = pd.crosstab(index=src["position"],columns=src["bmi_class"],normalize="index")

px.bar(pos_bmi,title="Position vs BMI Class",labels={'value':"% of players"})
cntry = pd.DataFrame(src["country"].value_counts()).reset_index()

less_cnt = cntry.loc[cntry["country"]<(0.005*cntry["country"].sum()),"index"]



cnt_bmi = src[["country","bmi_class"]].copy()

cnt_bmi.loc[cnt_bmi["country"].isin(less_cnt),"country"] = "Others"

cnt_bmi = pd.crosstab(index=cnt_bmi["country"],columns=cnt_bmi["bmi_class"],normalize="index").copy()

px.bar(cnt_bmi,title="Position vs BMI Class",labels={'value':"% of players"})