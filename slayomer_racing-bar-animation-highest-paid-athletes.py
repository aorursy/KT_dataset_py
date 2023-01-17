import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="darkgrid")

plt.style.use("seaborn-pastel")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel("/kaggle/input/yearly-highest-paid-100-athletes-forbes-20122019/Forbes Athlete List 2012-2019.xlsx")

df.head(10)
df.info()
df.Rank = df.Rank.apply(lambda x: int(x.split("#")[1]) if type(x) == np.str else x)

df.Pay = df.Pay.apply(lambda x: float(x.split(" ")[0].split("$")[1]))

df.Endorsements = df.Endorsements.apply(lambda x: float(x.split(" ")[0].split("$")[1]))

df["Salary/Winnings"].replace("-",'$nan M',inplace=True)

df["Salary/Winnings"] = df["Salary/Winnings"].apply(lambda x: float(x.split(" ")[0].split("$")[1]))

df.Sport.replace({"Soccer":"Football",

                  "Football":"American Football",

                 "Mixed Martial Arts":"MMA",

                 "Auto racing":"Racing",

                  "Auto Racing":"Racing",

                  "Basketbal":"Basketball",

                 },inplace=True)



df.columns = ['Rank', 'Name', 'Pay', 'Salary_Winnings', 'Endorsements', 'Sport', 'Year']

df.head(10)
messi = df[df.Name == "Lionel Messi"].sort_values("Year")

messi
sns.barplot(data=messi,x="Pay",y="Year",orient="h")

plt.title("Messi's Pay Progress")

plt.show()
df.groupby("Name").first()["Sport"].value_counts().plot(kind="pie",autopct="%.0f%%",figsize=(8,8),wedgeprops=dict(width=0.4),pctdistance=0.8)

plt.ylabel(None)

plt.title("Breakdown of Athletes by Sport",fontweight="bold")

plt.show()
sports = df.groupby("Sport").agg(

    total_pay = ("Pay","sum"),

    no_of_players = ("Name","count")

)



sports["pay_per_player"] = sports.total_pay/sports.no_of_players

sports.sort_values("pay_per_player",ascending=False)
df.Year = pd.to_datetime(df.Year,format="%Y")

df.dtypes
racing_bar_data = df.pivot_table(values="Pay",index="Year",columns="Name")

racing_bar_data.cumsum()
racing_bar_data.columns[racing_bar_data.isnull().sum() == 0]
racing_bar_filled = racing_bar_data.interpolate(method="linear").fillna(method="bfill")

racing_bar_filled
racing_bar_filled = racing_bar_filled.cumsum()

racing_bar_filled
racing_bar_filled = racing_bar_filled.resample("1D").interpolate(method="linear")[::7]
racing_bar_filled["Lionel Messi"].plot(marker=".",figsize=(12,4))

plt.ylabel("Cumulative Pay (Million USD)")

plt.show()
from matplotlib.animation import FuncAnimation, FFMpegWriter



selected  = racing_bar_filled.iloc[-1,:].sort_values(ascending=False)[:20].index

data = racing_bar_filled[selected].round()



fig,ax = plt.subplots(figsize=(9.3,7))

fig.subplots_adjust(left=0.18)

no_of_frames = data.shape[0] #Number of frames



#initiate the barplot with the first rows of the dataframe

bars = sns.barplot(y=data.columns,x=data.iloc[0,:],orient="h",ax=ax)

ax.set_xlim(0,1500)

txts = [ax.text(0,i,0,va="center") for i in range(data.shape[1])]

title_txt = ax.text(650,-1,"Date: ",fontsize=12)

ax.set_xlabel("Pay (Millions USD)")

ax.set_ylabel(None)



def animate(i):

#     print(f"i={i}/{no_of_frames}")

    #get i'th row of data 

    y = data.iloc[i,:]

    

    #update title of the barplot axis

    title_txt.set_text(f"Date: {str(data.index[i].date())}")

    

    #update elements in both plots

    for j, b, in enumerate(bars.patches):

        #update each bar's height

        b.set_width(y[j])

        

        #update text for each bar (optional)

        txts[j].set_text(f"${y[j].astype(int)}M")

        txts[j].set_x(y[j])



anim=FuncAnimation(fig,animate,repeat=False,frames=no_of_frames,interval=1,blit=False)

anim.save('athletes.gif', writer='imagemagick', fps=120)

plt.close(fig)