#First of all , I will import seaborn librariy and load tips dataset.



import seaborn as sns

tips = sns.load_dataset('tips')

df = tips.copy()

df.head()
# some statistic indicators



df.describe().T
df["sex"].value_counts()
df["smoker"].value_counts()
df["day"].value_counts()
df["time"].value_counts()
sns.boxplot(x=df["total_bill"]);
# We can also change the direction of the graph.



sns.boxplot(x=df["total_bill"],orient = "v");
# Which days restaurant make more money?



sns.boxplot(x="day", y="total_bill",data = df);
df["day"].value_counts()
# Which days survers make more money?



sns.boxplot(x="day", y="tip",data = df);
# And which part of the day restaurant make more money?



sns.boxplot(x="time", y="total_bill",data = df);
# What is relation between 'group size' and 'total bill' ?



sns.boxplot(x="size", y="total_bill",data = df);
# And Which gender pays the bill by days?



sns.boxplot(x="day", y="total_bill",hue = "sex", data = df);
sns.catplot(y="total_bill", kind = "violin",data= df);
sns.boxplot(x=df["total_bill"]);
# Let's cross 'day' and 'total_bill'



sns.catplot(x="day", y="total_bill", kind = "violin",data= df);
sns.boxplot(x="day", y="total_bill",data = df);
# And now cross 3 variable:  'day','total bill' and 'sex'



sns.catplot(x="day", y="total_bill", hue="sex",kind = "violin",data= df);
sns.scatterplot(x="total_bill", y="tip",data=df);
# Now we cross 3 variables: 'tip', 'time' and 'total_bill'



sns.scatterplot(x="total_bill", y="tip", hue = "time", data =df);
# Cross 'tip', 'day' and 'total_bill'



sns.scatterplot(x="total_bill", y="tip", hue = "day", style ="time", data =df);
# Now we cross 4 variables: 'tip', 'time','total_bill' and 'day'



sns.scatterplot(x="total_bill", y="tip", hue = "time", style ="day", data =df);
# And explore new vizulation methods.



sns.scatterplot(x="total_bill", y="tip",hue = "size", size = "size", data =df);
import matplotlib.pyplot as plt
sns.lmplot(x="total_bill", y="tip",data=df);
sns.lmplot(x="total_bill", y="tip",hue ="smoker" , data=df);
sns.lmplot(x="total_bill", y="tip",hue ="smoker", col = "time" , data=df);
sns.lmplot(x="total_bill", y="tip",hue ="smoker", col = "time" ,row = "sex", data=df);