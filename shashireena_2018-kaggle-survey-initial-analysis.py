#import python libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycountry
import seaborn as sns
import matplotlib_venn as vn
from matplotlib_venn import venn2
from matplotlib_venn import venn3
from IPython.display import display,HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import warnings
warnings.filterwarnings("ignore")
# import data set
dsfreeformrsp=pd.read_csv("../input/freeFormResponses.csv",header=1)
dsmultiplersp=pd.read_csv("../input/multipleChoiceResponses.csv",header=1)

filename={"freeFormResponses.csv":dsfreeformrsp,"multipleChoiceResponses.csv":dsmultiplersp}

def filestruc(filedic):
    datastat=pd.DataFrame()
    for i in filename.keys():
        row=pd.DataFrame({i:list(filename[i].shape)}).transpose()
        datastat=datastat.append(row)
    datastat.columns=["NoOfRows","NoOfColumns"]
    return datastat
    
filestruc(filename)
display(HTML(dsfreeformrsp.head(5).to_html()))
# Sample records: multipleChoiceResponses.csv
display(HTML(dsmultiplersp.head(5).to_html()))
TotalRec=list(dsmultiplersp.shape)[0]
dataset=pd.DataFrame(dsmultiplersp.transpose().count(axis=1))
dataset["NullRec%"]=round(((TotalRec-dataset[0])/TotalRec)*100,2)
dataset.columns=["CNT_NotNull","NullRec%"]
dataset=dataset.sort_values(by=["CNT_NotNull","NullRec%"],ascending=False).reset_index(drop=False).style.set_properties(align="left")
dataset

TotalRec=list(dsfreeformrsp.shape)[0]
dataset=pd.DataFrame(dsfreeformrsp.transpose().count(axis=1))
dataset["NullRec%"]=round(((TotalRec-dataset[0])/TotalRec)*100,2)
dataset.columns=["CNT_NotNull","NullRec%"]
dataset=dataset.sort_values(by=["CNT_NotNull","NullRec%"],ascending=False).reset_index(drop=False).style.set_properties(align="left")
dataset
# Kaggle participants distribution per country

mapping={country.name: country.alpha_2 for country in pycountry.countries}
CntryUser=dsmultiplersp.groupby(["In which country do you currently reside?"])["Duration (in seconds)"].count().reset_index()
CntryUser.columns=["CountryName","CNT"]

CntryUser.CountryName=CntryUser.CountryName.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
CntryUser.CountryName=CntryUser.CountryName.replace('United States of America','United States')
CntryUser.CountryName=CntryUser.CountryName.replace('Iran, Islamic Republic of...', 'Iran')

CntryUser["ISO_CntryCode"]=np.nan
for i,value in enumerate(CntryUser.CountryName):
    if value in list(mapping.keys()):
        CntryUser["ISO_CntryCode"][i]=mapping[value]
    else:
        CntryUser["ISO_CntryCode"][i]=None



#Ploty - World map

data=[dict(
    type='choropleth',
    locations = CntryUser.CountryName,
    locationmode = 'country names',
    z = CntryUser.CNT,
    colorscale = [[0,"rgb(240,230,140)"],[0.35,"rgb(255,215,0)"],[0.5,"rgb(255,165,0)"],\
            [0.7,"rgb(255,215,0)"],[1,"rgb(210,105,30)"]],
    autocolorscale = False,
    marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
    colorbar = dict(
            autotick = False,
            title = 'Kaggle<br>UserCount')
    )    
]

layout={
        "title":"Kaggle Users distribution per Country",
        "geo" : dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'))
        }
            
fig = dict(data=data, layout=layout)
iplot( fig, validate=False)
edusubset=dsmultiplersp[["What is your gender? - Selected Choice","What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"]].sort_values(by=["What is your gender? - Selected Choice"])
edusubset.columns=["gender","highestedulevel"]
edusubset.highestedulevel.replace("No formal education past high school",'High school',inplace=True)
edusubset.highestedulevel.replace("Some college/university study without earning a bachelor’s degree",'Bachelor’s degree dropout',inplace=True)
edusubset=edusubset.reset_index(drop=True)
edusubset=edusubset.groupby(["highestedulevel","gender"])["gender"].count().to_frame(name="Rec_cnt").reset_index()

fig,ax=plt.subplots(2,1,figsize=(10,12))

#Pie Plot - data scientist Gender distribution
dataset=edusubset.groupby(["gender"])["Rec_cnt"].sum().to_frame(name="TotCnt").reset_index()
dataset["%developer"]=round((dataset["TotCnt"]/list(dsmultiplersp.shape)[0])*100,2)
explode = (0, 0, 0.5, 0.2) 
ax[0].pie(dataset["%developer"],labels=dataset["gender"],shadow=True,autopct='%1.1f%%',explode = explode )
ax[0].axis('equal')
ax[0].set_title("Data scientist Gender distribution ")


#Bar Plot - data scientist Educational analysis
ax[1].patch.set_facecolor('gainsboro')
ax[1].xaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)
ax[1].yaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)

Male=ax[1].bar(data=edusubset,x=edusubset.highestedulevel.unique(),height=edusubset[edusubset.gender=="Male"].Rec_cnt,label='Male')
female=ax[1].bar(data=edusubset,x=edusubset.highestedulevel.unique(),height=edusubset[edusubset.gender=="Female"].Rec_cnt,bottom=np.array(edusubset[edusubset.gender=="Male"].Rec_cnt),label='Female')

for label in ax[1].xaxis.get_ticklabels():
    label.set_color('red')
    label.set_rotation(90)

for label in ax[1].yaxis.get_ticklabels():
    label.set_color('green')

# remove Axes border
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_visible(False)
ax[1].spines["bottom"].set_visible(False)

ax[1].set_title("Data scientist Educational analysis")
ax[1].set_ylabel("Count - Male vs Female")

ax[1].legend()

for bar in ax[1].patches:
    w, h = bar.get_width(), bar.get_height()
    if h<150:
        h=h+1000
    plt.text(bar.get_x() + w/2, bar.get_y() + h/2, round(bar.get_height(),1) ,ha="center",va="center")
plt.show()
    
roleanalysis=dsmultiplersp[["Select the title most similar to your current role (or most recent title if retired): - Selected Choice","What is your gender? - Selected Choice"]]
roleanalysis.columns=["CurrentWorkRole","Gender"]
roleanalysis=roleanalysis.groupby(["CurrentWorkRole"])["Gender"].value_counts().to_frame().unstack()
roleanalysis.columns=roleanalysis.columns.droplevel()
roleanalysis=roleanalysis.reset_index()

g=sns.PairGrid(roleanalysis.sort_values(by=["Male","Female"],ascending=False),x_vars=list(roleanalysis.columns)[1:],y_vars="CurrentWorkRole")
g.fig.set_size_inches(16,10)
g.map(sns.stripplot)
g.set(xlim=(-200,4200),xlabel="",ylabel="")

titles=["Female","Male","Prefer not to say", "Prefer to self-describe"]

for ax,title in zip(g.axes.flat,titles):
    ax.set(title=title)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)

agesector=dsmultiplersp[["What is your gender? - Selected Choice","What is your age (# years)?","Select the title most similar to your current role (or most recent title if retired): - Selected Choice","In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice"]]
agesector.columns=["gender","age_group","current_role","industry"]



fig,ax=plt.subplots(2,1,figsize=(8,15))

#*************************Age Group vs Professional role plot*******************
rect=sns.scatterplot(x="age_group",y="current_role",hue="gender",data=agesector,ax=ax[0])

#remove splines
rect.spines["top"].set_visible(False)
rect.spines["right"].set_visible(False)
rect.spines["bottom"].set_visible(False)
rect.spines["left"].set_visible(False)

rect.set_title("Age Group vs Professional Role",color="red",fontsize=14)
rect.set_xlabel("Age Group",color="green",fontsize=12)
rect.set_ylabel("Current Role",color="green",fontsize=12)

rect.patch.set_facecolor('gainsboro')
rect.xaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)
rect.yaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)

rect.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

#*************************Age group analysis ***************************************

g= agesector[(agesector.gender=="Male")|(agesector.gender=="Female")].groupby(["age_group","gender"])["gender"].count().to_frame(name="Rec_cnt").reset_index()
male=ax[1].bar(x=g[g.gender=="Male"].age_group,height=g[g.gender=="Male"].Rec_cnt,label='Male')
female=ax[1].bar(x=g[g.gender=="Female"].age_group,height=g[g.gender=="Female"].Rec_cnt,bottom=g[g.gender=="Male"].Rec_cnt,label='Female')

ax[1].set_title("Kagglers Age Group distribution",fontsize=14,color="red")
ax[1].set_xlabel("Age Group",fontsize=12,color="green")
ax[1].set_ylabel("Count",fontsize=12,color="green")

ax[1].patch.set_facecolor('gainsboro')
ax[1].xaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)
ax[1].yaxis.grid(True, linestyle='--', which='major',color='white', alpha=0.60)

ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["bottom"].set_visible(False)
ax[1].spines["left"].set_visible(False)
ax[1].legend()

for bar in ax[1].patches:
    w, h = bar.get_width(), bar.get_height()
    if h<150:
        h=h+800
    plt.text(bar.get_x() + w/2, bar.get_y() + h/2, round(bar.get_height(),1) ,ha="center",va="center")


#***********Sector analysis************************************************

z=agesector.industry.value_counts().to_frame(name="rec_cnt").reset_index()
z.columns=["Sector","Rec_cnt"]

sns.set()
fig,ax=plt.subplots(figsize=(7,9))
ax=sns.barplot(y="Sector",x="Rec_cnt",data=z)
sns.despine(left=True,bottom=True,right=True,top=True)

ax.set_xlabel("Count",color="green")
ax.set_ylabel("Sector",color="green")
plt.show()
sector=dsmultiplersp[["Select the title most similar to your current role (or most recent title if retired): - Selected Choice","In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice"]]
sector.columns=["WorkRole","Industry"]
sector.dropna(inplace=True)
sector=sector.groupby(["WorkRole","Industry"])["Industry"].count().to_frame("Rec_cnt").unstack()
sector.columns=sector.columns.droplevel()
fig,ax=plt.subplots(figsize=(18,10))
hmap=sns.heatmap(sector,annot=True,ax=ax,fmt='.1f')
hmap.set_xlabel("Industry",color="green",fontsize=12)
hmap.set_ylabel("WorkRole",color="green",fontsize=12)
plt.show()
IDEtool=dsmultiplersp[["Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Jupyter/IPython",\
             "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - RStudio",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - PyCharm",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio Code",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - nteract",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Atom",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - MATLAB",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Notepad++",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Sublime Text",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Vim",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - IntelliJ",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Spyder",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - None",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Other"]]

IDEtool.columns=["Jupyter/IPython","RStudio","PyCharm","Visual Studio Code","nteract","Atom","MATLAB","Visual Studio","Notepad++","Sublime Text","Vim","IntelliJ","Spyder","None",'Other']
IDEtool=IDEtool.fillna(0).replace('[^\d]',1, regex=True)
IDE=IDEtool.apply(np.sum,axis=0).to_frame().reset_index()
IDE.columns=["IDEname","recnt"]
IDE=IDE.sort_values("recnt",ascending=False).reset_index(drop=True)

fig,ax=plt.subplots(figsize=(14,10))
ax=sns.barplot(x="recnt",y="IDEname",data=IDE)

ax.set_title("2018 Kaggle survey IDE use",color="red",fontsize=15)
ax.set_xlabel("recnt",fontsize=14,color="green")
ax.set_ylabel("IDE Name",fontsize=14,color="green")


for bar in ax.patches:
    w, h = bar.get_width(), bar.get_height()
    annot=str(round((w/list(dsmultiplersp.shape)[0])*100,1))+"%"
    plt.text(bar.get_x() + w/2, bar.get_y() + h/2,annot,ha="center",va="center",color="w")
    

fig,ax=plt.subplots(figsize=(6,6))
ax=venn3(subsets=(
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==1)&(IDEtool["RStudio"]==0)&(IDEtool["Notepad++"]==0)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==0)&(IDEtool["RStudio"]==1)&(IDEtool["Notepad++"]==0)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==1)&(IDEtool["RStudio"]==1)&(IDEtool["Notepad++"]==0)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==0)&(IDEtool["RStudio"]==0)&(IDEtool["Notepad++"]==1)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==1)&(IDEtool["RStudio"]==0)&(IDEtool["Notepad++"]==1)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==0)&(IDEtool["RStudio"]==1)&(IDEtool["Notepad++"]==1)]),
len(IDEtool.loc[(IDEtool["Jupyter/IPython"]==1)&(IDEtool["RStudio"]==1)&(IDEtool["Notepad++"]==1)])),
set_labels=('Jupyter', 'RStudio', 'Notepad++'))
ax.get_patch_by_id('100').set_color('red')
ax.get_patch_by_id('010').set_color('green')
ax.get_patch_by_id('001').set_color('blueviolet')
plt.title('Jupyter vs RStudio vs Notepad++ (All users)',color="red")
plt.tight_layout()

plt.show()
Temp_rolevsIDE=dsmultiplersp[["Select the title most similar to your current role (or most recent title if retired): - Selected Choice",
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Jupyter/IPython",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - RStudio",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - PyCharm",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio Code",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - nteract",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Atom",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - MATLAB",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Visual Studio",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Notepad++",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Sublime Text",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Vim",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - IntelliJ",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Spyder",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - None",\
              "Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Other"]]

Temp_rolevsIDE.columns=["Workrole","Jupyter/IPython","RStudio","PyCharm","Visual Studio Code","nteract","Atom","MATLAB","Visual Studio","Notepad++","Sublime Text","Vim","IntelliJ","Spyder","None",'Other']
rolevsIDE=Temp_rolevsIDE.melt(id_vars=["Workrole"]).dropna()
rolevsIDE=pd.crosstab(index=rolevsIDE["Workrole"],columns=rolevsIDE["value"])


fig,ax=plt.subplots(figsize=(18,10))
hmap=sns.heatmap(rolevsIDE,annot=True,ax=ax,fmt='.0f')
hmap.set_xlabel("Industry",color="green",fontsize=14)
hmap.set_ylabel("WorkRole",color="green",fontsize=14)
plt.show()
cntryedu=dsmultiplersp[["In which country do you currently reside?","Which best describes your undergraduate major? - Selected Choice"]]
cntryedu.columns=["Country","Undergraduate_major"]

cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Engineering (non-computer focused)","Engineering(non-computer)")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Computer science (software engineering, etc.)","Computer science")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Social sciences (anthropology, psychology, sociology, etc.)","Social sciences")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Information technology, networking, or system administration","IT/networking")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("A business discipline (accounting, economics, finance, etc.)","Business discipline")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Environmental science or geology","Environmental science/geology")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Medical or life sciences (biology, chemistry, medicine, etc.)","Medical/life sciences")
cntryedu.Undergraduate_major=cntryedu.Undergraduate_major.replace("Humanities (history, literature, philosophy, etc.)","Humanities")
cntryedu.Country=cntryedu.Country.replace("United States of America","USA")
cntryedu.Country=cntryedu.Country.replace("United Kingdom of Great Britain and Northern Ireland","UK/Ireland")
top10cntry=pd.crosstab(index=cntryedu["Country"],columns=cntryedu["Undergraduate_major"],margins=True, margins_name="Total").drop("Other",axis=0)

cmap = cmap=sns.diverging_palette(250, 20, as_cmap=True)

z=top10cntry.sort_values(by=["Total"],ascending=False).iloc[1:11,0:14]

round((z.div(z.iloc[:,13],axis='index')*100),1).iloc[:,0:13].T.sort_values(by=["USA"],ascending=False).\
style.background_gradient(cmap, axis=1)

hostedbook=dsmultiplersp[["Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Kaggle Kernels",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Google Colab",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Azure Notebook",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Domino Datalab",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Google Cloud Datalab",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Paperspace",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Floydhub",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Crestle",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - JupyterHub/Binder",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - None",\
                         "Which of the following hosted notebooks have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Other"]]
hostedbook.columns=["Kaggle Kernels","Google Colab","Azure Notebook","Domino Datalab","Google Cloud Datalab","Paperspace","Floydhub","Crestle","JupyterHub/Binder","None","Other"]


cloudplt=dsmultiplersp[["Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Google Cloud Platform (GCP)",\
                        "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Amazon Web Services (AWS)",\
                        "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Microsoft Azure",\
                       "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - IBM Cloud",\
                       "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Alibaba Cloud",\
                       "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - I have not used any cloud providers",\
                       "Which of the following cloud computing services have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Other"]]

cloudplt.columns=["Google Cloud Platform (GCP)","Amazon Web Services (AWS)","Microsoft Azure","IBM Cloud","Alibaba Cloud","Not used Cloud","Other"]
hostedbook.stat=hostedbook.fillna(0).replace("[^\d]",1,regex=True).apply(np.sum).to_frame(name="Rec_cnt").reset_index()
hostedbook.stat.columns=["hostednotebook","Rec_cnt"]

hostedbook.stat=hostedbook.stat.sort_values(by=["Rec_cnt"],ascending=False).reset_index(drop=True)
sns.set()
fig,ax=plt.subplots(2,1,figsize=(8,15))
plt.subplots_adjust(hspace=0.6)
hostbook=sns.barplot(x="hostednotebook",y="Rec_cnt",data=hostedbook.stat,ax=ax[0])

hostbook.set_xlabel("Hosted Notebook")
hostbook.set_ylabel("No. of Users")
hostbook.set_title("Hosted notebook Kaggle survey use",color="red",fontsize=15)
for label in hostbook.xaxis.get_ticklabels():
    label.set_rotation(90)
    label.set_fontsize(11)

cloudplt.stat=cloudplt.fillna(0).replace("[^\d]",1,regex=True).apply(np.sum).to_frame(name="tot_cnt").reset_index()
cloudplt.stat.columns=["Cloudplt","tot_cnt"]
cloudplt.stat=cloudplt.stat.sort_values(by=["tot_cnt"],ascending=False).reset_index(drop=True)

cloudven=sns.barplot(y="Cloudplt",x="tot_cnt",data=cloudplt.stat,ax=ax[1])
cloudven.set_xlabel("No. of Users")
cloudven.set_ylabel("Cloud Platform")
cloudven.set_title("Cloud platform used by Kaggle users",color="red",fontsize=15)
plt.show()
