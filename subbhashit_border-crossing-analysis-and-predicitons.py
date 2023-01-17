import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use("classic")
data=pd.read_csv("../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv")
data.head()
data.sort_values('Value',inplace=True)
data.head()
data["Port Name"].value_counts().tail().plot(kind='bar',color=['c','y','r'],figsize=(15,10))
data.State.value_counts().plot(kind='pie',figsize=(15,10))
def type_of(x):
    if "Rail" in x:
        return "Rail"
    elif "Truck" in x:
        return "Truck"
    elif "Buses" in x:
        return "Buses"
    elif "Train" in x:
        return "Train"
    elif "Personal" in x:
        return "Personal"
    else:
        return "Other"
data.Measure=data.Measure.apply(type_of)
data.Measure.value_counts().plot(kind="pie",explode=(0,0,0.1,0,0.1,0),shadow=True,figsize=(15,10))
data.Border.value_counts().plot(color='r',figsize=(15,10))
data.Border.value_counts().plot(kind="bar",figsize=(15,10))
data["Port Name"].shape
top=data["Port Name"].head(200)
a=pd.crosstab(top,data.Border)
a.plot.bar(stacked=True,color=['r','g'],figsize=(15,15))
data['Date'] = pd.to_datetime(data['Date'])
data.head()
data['Date'].dt.year.value_counts().plot(kind="bar",cmap="rainbow",figsize=(15,10))
data.info()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data["Port Name"]=lab.fit_transform(data["Port Name"])
data["State"]=data["State"]
data["Border"]=data["Border"]
data["Measure"]=data["Measure"]
data.head()