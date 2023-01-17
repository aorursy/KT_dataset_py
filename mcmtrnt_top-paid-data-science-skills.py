import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



#Load in data

DS = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', delimiter = ',')

#DS.head()

DS['Q3'] = DS['Q3'].astype('category')

DS['Q4'] = DS['Q4'].astype('category')

DS['Q2'] = DS['Q2'].astype('category')

DS['Q1'] = DS['Q1'].astype('category')

DS['Q5'] = DS['Q5'].astype('category')

DS['Q10'] = DS['Q10'].astype('category')

DS['Q14'] = DS['Q14'].astype('category')

HighestPaid = DS[(DS.Q10 == "> $500,000") | (DS.Q10 == "100,000-124,999") | (DS.Q10 == "125,000-149,999") | (DS.Q10 == "150,000-199,999") | (DS.Q10 == "200,000-249,999") | (DS.Q10 == "250,000-299,999") | (DS.Q10 == "300,000-500,000")]

#HighestPaid.head()
GCP = HighestPaid['Q29_Part_1'].count() 

AWS = HighestPaid['Q29_Part_2'].count() 

MicrosoftAzure = HighestPaid['Q29_Part_3'].count() 

IBMCloud = HighestPaid['Q29_Part_4'].count() 

AlibabaCloud = HighestPaid['Q29_Part_5'].count() 

SalesforceCloud = HighestPaid['Q29_Part_6'].count() 

OracleCloud = HighestPaid['Q29_Part_7'].count() 

SAPCloud = HighestPaid['Q29_Part_8'].count() 

VMwareCloud = HighestPaid['Q29_Part_9'].count() 

RedHatCloud = HighestPaid['Q29_Part_10'].count() 



nsum = GCP + AWS + MicrosoftAzure + IBMCloud + AlibabaCloud + SalesforceCloud + OracleCloud + SAPCloud + VMwareCloud + RedHatCloud



GCP = GCP / nsum

AWS = AWS / nsum

MicrosoftAzure = MicrosoftAzure / nsum

IBMCloud = IBMCloud / nsum

AlibabaCloud = AlibabaCloud / nsum

SalesforceCloud = SalesforceCloud /nsum 

OracleCloud = OracleCloud / nsum

SAPCloud = SAPCloud / nsum

VMwareCloud = VMwareCloud / nsum

RedHatCloud = RedHatCloud / nsum



############



GCP1 = DS['Q29_Part_1'].count() 

AWS1 = DS['Q29_Part_2'].count() 

MicrosoftAzure1 = DS['Q29_Part_3'].count() 

IBMCloud1 = DS['Q29_Part_4'].count() 

AlibabaCloud1 = DS['Q29_Part_5'].count() 

SalesforceCloud1 = DS['Q29_Part_6'].count() 

OracleCloud1 = DS['Q29_Part_7'].count() 

SAPCloud1 = DS['Q29_Part_8'].count() 

VMwareCloud1 = DS['Q29_Part_9'].count() 

RedHatCloud1 = DS['Q29_Part_10'].count() 



nsum = GCP1 + AWS1 + MicrosoftAzure1 + IBMCloud1 + AlibabaCloud1 + SalesforceCloud1 + OracleCloud1 + SAPCloud1 + VMwareCloud1 + RedHatCloud1



GCP1 = GCP1 / nsum

AWS1 = AWS1 / nsum

MicrosoftAzure1 = MicrosoftAzure1 / nsum

IBMCloud1 = IBMCloud1 / nsum

AlibabaCloud1 = AlibabaCloud1 / nsum

SalesforceCloud1 = SalesforceCloud1 /nsum 

OracleCloud1 = OracleCloud1 / nsum

SAPCloud1 = SAPCloud1 / nsum

VMwareCloud1 = VMwareCloud1 / nsum

RedHatCloud1 = RedHatCloud1 / nsum



plt.style.use('ggplot')



x = ['Google Cloud Platform GCP', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'IBM Cloud', 'Alibaba Cloud', 'Salesforce Cloud', 'Oracle Cloud', 'SAP Cloud', 'VMware Cloud', 'Red Hat Cloud']

energy = [GCP, AWS, MicrosoftAzure , IBMCloud, AlibabaCloud , SalesforceCloud, OracleCloud , SAPCloud, VMwareCloud, RedHatCloud]

energy1 = [GCP1, AWS1, MicrosoftAzure1 , IBMCloud1, AlibabaCloud1 , SalesforceCloud1, OracleCloud1 , SAPCloud1, VMwareCloud1, RedHatCloud1]



N = 10

ind = np.arange(N) 

width = 0.35  



#x_pos = [i for i, _ in enumerate(x)]



plt.bar(ind + width, energy, width, color='Blue')

plt.bar(ind, energy1, width, color='Green')

plt.xlabel("Platform")

plt.ylabel("Percent of People")

plt.title("Which of the following cloud computing platforms do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
Scikitlearn  = DS['Q28_Part_1'].count() 

TensorFlow = DS['Q28_Part_2'].count() 

Keras  = DS['Q28_Part_3'].count() 

RandomForest = DS['Q28_Part_4'].count() 

Xgboost = DS['Q28_Part_5'].count() 

PyTorch = DS['Q28_Part_6'].count() 

Caret = DS['Q28_Part_7'].count() 

LightGBM = DS['Q28_Part_8'].count() 

SparkMLib = DS['Q28_Part_9'].count() 

Fastai = DS['Q28_Part_10'].count() 



nsum = Scikitlearn + TensorFlow + Keras + RandomForest + Xgboost + PyTorch + Caret + LightGBM + SparkMLib + Fastai



Scikitlearn  = Scikitlearn / nsum

TensorFlow = TensorFlow / nsum

Keras  = Keras/ nsum

RandomForest = RandomForest/ nsum

Xgboost = Xgboost / nsum

PyTorch = PyTorch/ nsum

Caret = Caret / nsum 

LightGBM = LightGBM / nsum 

SparkMLib = SparkMLib / nsum 

Fastai = Fastai / nsum



########################



Scikitlearn1  = HighestPaid['Q28_Part_1'].count() 

TensorFlow1 = HighestPaid['Q28_Part_2'].count() 

Keras1  = HighestPaid['Q28_Part_3'].count() 

RandomForest1 = HighestPaid['Q28_Part_4'].count() 

Xgboost1 = HighestPaid['Q28_Part_5'].count() 

PyTorch1 = HighestPaid['Q28_Part_6'].count() 

Caret1 = HighestPaid['Q28_Part_7'].count() 

LightGBM1 = HighestPaid['Q28_Part_8'].count() 

SparkMLib1 = HighestPaid['Q28_Part_9'].count() 

Fastai1 = HighestPaid['Q28_Part_10'].count() 



nsum1 = Scikitlearn1 + TensorFlow1 + Keras1 + RandomForest1 + Xgboost1 + PyTorch1 + Caret1 + LightGBM1 + SparkMLib1 + Fastai1



Scikitlearn1  = Scikitlearn1 / nsum1

TensorFlow1 = TensorFlow1 / nsum1

Keras1  = Keras1 / nsum1

RandomForest1 = RandomForest1 / nsum1

Xgboost1 = Xgboost1 / nsum1

PyTorch1 = PyTorch1 / nsum1

Caret1 = Caret1 / nsum1

LightGBM1 = LightGBM1 / nsum1

SparkMLib1 = SparkMLib1 / nsum1

Fastai1 = Fastai1 / nsum1



plt.style.use('ggplot')



x = ['Scikit-learn', 'TensorFlow', 'Keras', 'RandomForest', 'Xgboost', 'PyTorch', 'Caret', 'LightGBM', 'Spark MLib', 'Fast.ai ']

energy = [Scikitlearn, TensorFlow , Keras, RandomForest, Xgboost, PyTorch, Caret, LightGBM, SparkMLib , Fastai ]

energy1 = [Scikitlearn1, TensorFlow1 , Keras1, RandomForest1, Xgboost1, PyTorch1, Caret1, LightGBM1, SparkMLib1, Fastai1]



N = 10

ind = np.arange(N) 

width = 0.35  



#x_pos = [i for i, _ in enumerate(x)]



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("Framework")

plt.ylabel("Percent of People")

plt.title("Which of the following machine learning frameworks do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
KaggleNotebooks = DS['Q17_Part_1'].count() 

GoogleColab  = DS['Q17_Part_2'].count() 

MicrosoftAzureNotebooks  = DS['Q17_Part_3'].count() 

GoogleNotebooks = DS['Q17_Part_4'].count() 

PaperspaceGradient  = DS['Q17_Part_5'].count() 

FloydHub = DS['Q17_Part_6'].count() 

BinderJupyterHub  = DS['Q17_Part_7'].count() 

IBMWatsonStudio = DS['Q17_Part_8'].count() 

CodeOcean  = DS['Q17_Part_9'].count() 

AWSNotebookProducts = DS['Q17_Part_10'].count() 



nsum = KaggleNotebooks + GoogleColab + MicrosoftAzureNotebooks + GoogleNotebooks + PaperspaceGradient + FloydHub + BinderJupyterHub + IBMWatsonStudio + CodeOcean + AWSNotebookProducts



KaggleNotebooks = KaggleNotebooks / nsum

GoogleColab  = GoogleColab / nsum

MicrosoftAzureNotebooks  =  MicrosoftAzureNotebooks / nsum

GoogleNotebooks =  GoogleNotebooks / nsum

PaperspaceGradient  = PaperspaceGradient / nsum

FloydHub = FloydHub / nsum

BinderJupyterHub  =  BinderJupyterHub / nsum

IBMWatsonStudio =  IBMWatsonStudio / nsum

CodeOcean  =  CodeOcean / nsum

AWSNotebookProducts =  AWSNotebookProducts / nsum



##############33



KaggleNotebooks1 = HighestPaid['Q17_Part_1'].count() 

GoogleColab1  = HighestPaid['Q17_Part_2'].count() 

MicrosoftAzureNotebooks1  = HighestPaid['Q17_Part_3'].count() 

GoogleNotebooks1 = HighestPaid['Q17_Part_4'].count() 

PaperspaceGradient1  = HighestPaid['Q17_Part_5'].count() 

FloydHub1 = HighestPaid['Q17_Part_6'].count() 

BinderJupyterHub1  = HighestPaid['Q17_Part_7'].count() 

IBMWatsonStudio1 = HighestPaid['Q17_Part_8'].count() 

CodeOcean1  = HighestPaid['Q17_Part_9'].count() 

AWSNotebookProducts1 = HighestPaid['Q17_Part_10'].count() 



nsum1 = KaggleNotebooks1 + GoogleColab1 + MicrosoftAzureNotebooks1 + GoogleNotebooks1 + PaperspaceGradient1 + FloydHub1 + BinderJupyterHub1 + IBMWatsonStudio1 + CodeOcean1 + AWSNotebookProducts1



KaggleNotebooks1 = KaggleNotebooks1 / nsum1

GoogleColab1  = GoogleColab1 / nsum1

MicrosoftAzureNotebooks1  =  MicrosoftAzureNotebooks1 / nsum1

GoogleNotebooks1 =  GoogleNotebooks1 / nsum1

PaperspaceGradient1  = PaperspaceGradient1 / nsum1

FloydHub1 = FloydHub1 / nsum1

BinderJupyterHub1  =  BinderJupyterHub1 / nsum1

IBMWatsonStudio1 =  IBMWatsonStudio1 / nsum1

CodeOcean1  =  CodeOcean1 / nsum1

AWSNotebookProducts1 =  AWSNotebookProducts1 / nsum1



plt.style.use('ggplot')



x = ['Kaggle Notebooks', 'Google Colab ', 'Microsoft Azure Notebooks ', 'Google Cloud Notebook Products', 'Paperspace / Gradient ', 'FloydHub', 'Binder / JupyterHub', 'IBM Watson Studio', 'Code Ocean', 'AWS Notebook Products']

energy = [KaggleNotebooks, GoogleColab, MicrosoftAzureNotebooks, GoogleNotebooks, PaperspaceGradient, FloydHub, BinderJupyterHub, IBMWatsonStudio, CodeOcean, AWSNotebookProducts]

energy1 = [KaggleNotebooks1, GoogleColab1, MicrosoftAzureNotebooks1, GoogleNotebooks1, PaperspaceGradient1, FloydHub1, BinderJupyterHub1, IBMWatsonStudio1, CodeOcean1, AWSNotebookProducts1]



N = 10

ind = np.arange(N) 

width = 0.35  



#x_pos = [i for i, _ in enumerate(x)]



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("Notebook")

plt.ylabel("Percent of People")

plt.title("Which of the following hosted notebook products do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
Udacity = DS['Q13_Part_1'].count() 

Coursera = DS['Q13_Part_2'].count() 

edX = DS['Q13_Part_3'].count() 

DataCamp = DS['Q13_Part_4'].count() 

DataQuest = DS['Q13_Part_5'].count() 

KaggleCourses = DS['Q13_Part_6'].count() 

Fastai = DS['Q13_Part_7'].count() 

Udemy = DS['Q13_Part_8'].count() 

LinkedinLearning = DS['Q13_Part_9'].count() 

UniversityCourses = DS['Q13_Part_10'].count() 



nsum = Udacity + Coursera + edX + DataCamp + DataQuest + KaggleCourses + Fastai + Udemy + LinkedinLearning + UniversityCourses



Udacity = Udacity / nsum

Coursera = Coursera / nsum

edX = edX / nsum

DataCamp = DataCamp / nsum

DataQuest = DataQuest / nsum

KaggleCourses = KaggleCourses / nsum

Fastai = Fastai / nsum

Udemy = Udemy / nsum

LinkedinLearning = LinkedinLearning / nsum

UniversityCourses = UniversityCourses / nsum



##############



Udacity1 = HighestPaid['Q13_Part_1'].count() 

Coursera1 = HighestPaid['Q13_Part_2'].count() 

edX1 = HighestPaid['Q13_Part_3'].count() 

DataCamp1 = HighestPaid['Q13_Part_4'].count() 

DataQuest1 = HighestPaid['Q13_Part_5'].count() 

KaggleCourses1 = HighestPaid['Q13_Part_6'].count() 

Fastai1 = HighestPaid['Q13_Part_7'].count() 

Udemy1 = HighestPaid['Q13_Part_8'].count() 

LinkedinLearning1 = HighestPaid['Q13_Part_9'].count() 

UniversityCourses1 = HighestPaid['Q13_Part_10'].count() 



nsum1 = Udacity1 + Coursera1 + edX1 + DataCamp1 + DataQuest1 + KaggleCourses1 + Fastai1 + Udemy1 + LinkedinLearning1 + UniversityCourses1



Udacity1 = Udacity1 / nsum1

Coursera1 = Coursera1 / nsum1

edX1 = edX1 / nsum1

DataCamp1 = DataCamp1 / nsum1

DataQuest1 = DataQuest1 / nsum1

KaggleCourses1 = KaggleCourses1 / nsum1

Fastai1 = Fastai1 / nsum1

Udemy1 = Udemy1 / nsum1

LinkedinLearning1 = LinkedinLearning1 / nsum1

UniversityCourses1 = UniversityCourses1 / nsum1



plt.style.use('ggplot')



x = ['Udacity', 'Coursera', 'edX', 'DataCamp', 'DataQuest', 'Kaggle Courses', 'Fast.ai', 'Udemy', 'Linkedin Learning', 'University Courses']

energy = [Udacity, Coursera, edX, DataCamp, DataQuest, KaggleCourses, Fastai, Udemy, LinkedinLearning, UniversityCourses]

energy1 = [Udacity1, Coursera1, edX1, DataCamp1, DataQuest1, KaggleCourses1, Fastai1, Udemy1, LinkedinLearning1, UniversityCourses1]



N = 10

ind = np.arange(N) 

width = 0.35  



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("Platform")

plt.ylabel("Percent of People")

plt.title("On which platforms have Highly Paid Professionals begun or completed data science courses?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
Jupyter = DS['Q16_Part_1'].count() 

RStudio = DS['Q16_Part_2'].count() 

PyCharm = DS['Q16_Part_3'].count() 

Atom = DS['Q16_Part_4'].count() 

MATLAB = DS['Q16_Part_5'].count() 

VisualStudio = DS['Q16_Part_6'].count() 

Spyder = DS['Q16_Part_7'].count() 

VimEmacs = DS['Q16_Part_8'].count() 

NotePad = DS['Q16_Part_9'].count() 

SublimeText = DS['Q16_Part_10'].count() 



nsum = Jupyter + RStudio + PyCharm + Atom + MATLAB + VisualStudio + Spyder + VimEmacs + NotePad + SublimeText



Jupyter = Jupyter / nsum

RStudio = RStudio / nsum

PyCharm = PyCharm / nsum

Atom = Atom / nsum

MATLAB = MATLAB / nsum

VisualStudio = VisualStudio / nsum

Spyder = Spyder / nsum

VimEmacs = VimEmacs / nsum

NotePad = NotePad / nsum

SublimeText = SublimeText / nsum



##############



Jupyter1 = HighestPaid['Q16_Part_1'].count() 

RStudio1 = HighestPaid['Q16_Part_2'].count() 

PyCharm1 = HighestPaid['Q16_Part_3'].count() 

Atom1 = HighestPaid['Q16_Part_4'].count() 

MATLAB1 = HighestPaid['Q16_Part_5'].count() 

VisualStudio1 = HighestPaid['Q16_Part_6'].count() 

Spyder1 = HighestPaid['Q16_Part_7'].count() 

VimEmacs1 = HighestPaid['Q16_Part_8'].count() 

NotePad1 = HighestPaid['Q16_Part_9'].count() 

SublimeText1 = HighestPaid['Q16_Part_10'].count() 



nsum1 = Jupyter1 + RStudio1 + PyCharm1 + Atom1 + MATLAB1 + VisualStudio1 + Spyder1 + VimEmacs1 + NotePad1 + SublimeText1



Jupyter1 = Jupyter1 / nsum1

RStudio1 = RStudio1 / nsum1

PyCharm1 = PyCharm1 / nsum1

Atom1 = Atom1 / nsum1

MATLAB1 = MATLAB1 / nsum1

VisualStudio1 = VisualStudio1 / nsum1

Spyder1 = Spyder1 / nsum1

VimEmacs1 = VimEmacs1 / nsum1

NotePad1 = NotePad1 / nsum1

SublimeText1 = SublimeText1 / nsum1



plt.style.use('ggplot')



x = ['Jupyter', 'RStudio', 'PyCharm', 'Atom', 'MATLAB', 'Visual Studio', 'Spyder', 'Vim / Emacs', 'NotePad++', 'Sublime Text']

energy = [Jupyter, RStudio, PyCharm, Atom, MATLAB, VisualStudio, Spyder, VimEmacs, NotePad, SublimeText]

energy1 = [Jupyter1, RStudio1, PyCharm1, Atom1, MATLAB1, VisualStudio1, Spyder1, VimEmacs1, NotePad1, SublimeText1]



N = 10

ind = np.arange(N) 

width = 0.35  



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("IDE")

plt.ylabel("Percent of People")

plt.title("Which of the following IDE's do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
Python = DS['Q18_Part_1'].count() 

R  = DS['Q18_Part_2'].count() 

SQL  = DS['Q18_Part_3'].count() 

C = DS['Q18_Part_4'].count() 

Cplusplus = DS['Q18_Part_5'].count() 

Java = DS['Q18_Part_6'].count() 

Javascript  = DS['Q18_Part_7'].count() 

TypeScript = DS['Q18_Part_8'].count() 

Bash  = DS['Q18_Part_9'].count() 

MATLAB = DS['Q18_Part_10'].count() 



nsum = Python + R + SQL + C + Cplusplus + Java + Javascript + TypeScript + Bash + MATLAB



Python = Python / nsum

R  = R / nsum

SQL = SQL / nsum

C =  C / nsum

Cplusplus = Cplusplus / nsum

Java = Java / nsum

Javascript = Javascript / nsum

TypeScript = TypeScript / nsum

Bash = Bash / nsum

MATLAB = MATLAB / nsum



########



hPython = HighestPaid['Q18_Part_1'].count() 

hR  = HighestPaid['Q18_Part_2'].count() 

hSQL  = HighestPaid['Q18_Part_3'].count() 

hC = HighestPaid['Q18_Part_4'].count() 

hCplusplus = HighestPaid['Q18_Part_5'].count() 

hJava = HighestPaid['Q18_Part_6'].count() 

hJavascript  = HighestPaid['Q18_Part_7'].count() 

hTypeScript = HighestPaid['Q18_Part_8'].count() 

hBash  = HighestPaid['Q18_Part_9'].count() 

hMATLAB = HighestPaid['Q18_Part_10'].count() 



nsumh = hPython + hR + hSQL + hC + hCplusplus + hJava + hJavascript + hTypeScript + hBash + hMATLAB



hPython = hPython / nsumh

hR  = hR / nsumh

hSQL = hSQL / nsumh

hC =  hC / nsumh

hCplusplus = hCplusplus / nsumh

hJava = hJava / nsumh

hJavascript = hJavascript / nsumh

hTypeScript = hTypeScript / nsumh

hBash = hBash / nsumh

hMATLAB = hMATLAB / nsumh





plt.style.use('ggplot')



x = ['Python', 'R', 'SQL', 'C', 'C++', 'Java', 'Javascript', 'TypeScript', 'Bash', 'MATLAB']

energy = [Python, R, SQL, C, Cplusplus, Java, Javascript, TypeScript, Bash, MATLAB]

energy1 = [hPython, hR, hSQL, hC, hCplusplus, hJava, hJavascript, hTypeScript, hBash, hMATLAB]



N = 10

ind = np.arange(N) 

width = 0.35  



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("Language")

plt.ylabel("Percent of People")

plt.title("What programming languages do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
Ggplotggplot2 = DS['Q20_Part_1'].count() 

Matplotlib = DS['Q20_Part_2'].count() 

Altair  = DS['Q20_Part_3'].count() 

Shiny = DS['Q20_Part_4'].count() 

D3js = DS['Q20_Part_5'].count() 

PlotlyPlotlyExpress = DS['Q20_Part_6'].count() 

Bokeh = DS['Q20_Part_7'].count() 

Seaborn = DS['Q20_Part_8'].count() 

Geoplotlib = DS['Q20_Part_9'].count() 

LeafletFolium  = DS['Q20_Part_10'].count() 



nsum = Ggplotggplot2 + Matplotlib + Altair + Shiny + D3js + PlotlyPlotlyExpress + Bokeh + Seaborn + Geoplotlib + LeafletFolium



Ggplotggplot2 =  Ggplotggplot2 / nsum

Matplotlib =  Matplotlib / nsum

Altair = Altair / nsum

Shiny =  Shiny / nsum

D3js = D3js / nsum

PlotlyPlotlyExpress = PlotlyPlotlyExpress / nsum

Bokeh = Bokeh / nsum

Seaborn = Seaborn / nsum

Geoplotlib = Geoplotlib / nsum

LeafletFolium  = LeafletFolium / nsum



##################



hGgplotggplot2 = HighestPaid['Q20_Part_1'].count() 

hMatplotlib = HighestPaid['Q20_Part_2'].count() 

hAltair  = HighestPaid['Q20_Part_3'].count() 

hShiny = HighestPaid['Q20_Part_4'].count() 

hD3js = HighestPaid['Q20_Part_5'].count() 

hPlotlyPlotlyExpress = HighestPaid['Q20_Part_6'].count() 

hBokeh = HighestPaid['Q20_Part_7'].count() 

hSeaborn = HighestPaid['Q20_Part_8'].count() 

hGeoplotlib = HighestPaid['Q20_Part_9'].count() 

hLeafletFolium  = HighestPaid['Q20_Part_10'].count() 



nsumh = hGgplotggplot2 + hMatplotlib + hAltair + hShiny + hD3js + hPlotlyPlotlyExpress + hBokeh + hSeaborn + hGeoplotlib + hLeafletFolium



hGgplotggplot2 =  hGgplotggplot2 / nsumh

hMatplotlib =  hMatplotlib / nsumh

hAltair = hAltair / nsumh

hShiny =  hShiny / nsumh

hD3js = hD3js / nsumh

hPlotlyPlotlyExpress = hPlotlyPlotlyExpress / nsumh

hBokeh = hBokeh / nsumh

hSeaborn = hSeaborn / nsumh

hGeoplotlib = hGeoplotlib / nsumh

hLeafletFolium  = hLeafletFolium / nsumh





plt.style.use('ggplot')



x = ['Ggplot / ggplot2', 'Matplotlib', 'Altair', 'Shiny', 'D3.js', 'Plotly / Plotly Express', 'Bokeh', 'Seaborn', 'Geoplotlib ', 'Leaflet / Folium ']

energy = [Ggplotggplot2, Matplotlib , Altair, Shiny, D3js, PlotlyPlotlyExpress, Bokeh, Seaborn, Geoplotlib , LeafletFolium ]

energy1 = [hGgplotggplot2, hMatplotlib , hAltair, hShiny, hD3js, hPlotlyPlotlyExpress, hBokeh, hSeaborn, hGeoplotlib , hLeafletFolium ]



N = 10

ind = np.arange(N) 

width = 0.35  



plt.bar(ind, energy, width, color='Green')

plt.bar(ind + width, energy1, width, color='Blue')

plt.xlabel("Library")

plt.ylabel("Percent of People")

plt.title("What data visualization libraries or tools do you use on a regular basis?")



plt.xticks(ind, x, rotation=40, ha="right")



plt.show()
ax = sns.countplot(x="Q5", data=DS)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
ax = sns.countplot(x="Q5", data=HighestPaid)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
ax = sns.countplot(x="Q14", data=DS)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
ax = sns.countplot(x="Q14", data=HighestPaid)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
ax = sns.countplot(x="Q19", data=HighestPaid)



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()