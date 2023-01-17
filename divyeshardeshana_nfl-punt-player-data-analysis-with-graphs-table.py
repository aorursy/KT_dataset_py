import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.figure_factory as ff 
import  plotly.offline as py
import plotly.graph_objs as go 
from plotly.offline  import download_plotlyjs,init_notebook_mode,plot, iplot, plot
import cufflinks as cf
from plotly import tools 
py.init_notebook_mode(connected = True)
import cufflinks as cf 
cf.go_offline()
%matplotlib inline
Game_Data_Set = pd.read_csv('../input/game_data.csv')
Play_Information_Data_Set = pd.read_csv('../input/play_information.csv')
Player_Punt_Data_Set = pd.read_csv('../input/player_punt_data.csv')
Play_Player_Role_Data_Set = pd.read_csv('../input/play_player_role_data.csv')
Video_Review_Data_Set = pd.read_csv('../input/video_review.csv')
Game_Data_Set = Game_Data_Set.rename({
    'GameKey' : 'Game_Key',
    'HomeTeamCode' : 'Home_Team_Code',
    'VisitTeamCode' : 'Visit_Team_Code',
    'StadiumType' : 'Stadium_Type',
    'GameWeather' : 'Game_Weather',
    'OutdoorWeather' : 'Outdoor_Weather'
    }, axis='columns')
Play_Information_Data_Set = Play_Information_Data_Set.rename({
    'GameKey' : 'Game_Key',
    'PlayID' : 'Play_ID',
    'YardLine' : 'Yard_Line',
    'PlayDescription' : 'Play_Description',
    }, axis='columns')
Play_Player_Role_Data_Set = Play_Player_Role_Data_Set.rename({
    'GameKey' : 'Game_Key',
    'PlayID' : 'Play_ID'
    }, axis='columns')
Video_Review_Data_Set = Video_Review_Data_Set.rename({
    'GameKey' : 'Game_Key',
    'PlayID' : 'Play_ID'
    }, axis='columns')
Game_Data_Set['Stadium_Type'].fillna("Missing Stadium Type", inplace=True)
Game_Data_Set['Turf'].fillna("Missing Turf", inplace=True)
Game_Data_Set['Game_Weather'].fillna("Missing Weather", inplace=True)
Game_Data_Set['Temperature'].fillna("0.00", inplace=True)
Game_Data_Set['Outdoor_Weather'].fillna("Missing Outdoor Weather", inplace=True)
Video_Review_Data_Set['Primary_Partner_GSISID'].fillna("0.00", inplace=True)
Video_Review_Data_Set['Primary_Partner_Activity_Derived'].fillna("Missing Primary Partner Activity", inplace=True)
Video_Review_Data_Set['Friendly_Fire'].fillna("Missing Friendly Fire", inplace=True)
Video_Review_Data_Set.Season_Year = Video_Review_Data_Set.Season_Year.astype('int64')
Game_Data_Set.info()
Play_Information_Data_Set.info()
Player_Punt_Data_Set.info()
Play_Player_Role_Data_Set.info()
Video_Review_Data_Set.info()
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "AT&T", "Stadium"] = 'AT&T Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Bank of America", "Stadium"] = 'Bank of America Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "CenturyLink", "Stadium"] = 'CenturyLink Field'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "FirstEnergy", "Stadium"] = 'First Energy Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "FirstEnergy Stadium", "Stadium"] = 'First Energy Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "First Energy Stadium", "Stadium"] = 'First Energy Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Los  Angeles Memorial Coliseum", "Stadium"] = 'Los Angeles Memorial Coliseum'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Lucas Oil", "Stadium"] = 'Lucas Oil Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "M & T Bank Stadium", "Stadium"] = 'M&T Bank Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "M&T Stadium", "Stadium"] = 'M&T Bank Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "MetLife", "Stadium"] = 'MetLife Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Mercedes Benz-Superdome", "Stadium"] = 'Mercedes-Benz Superdome'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "NRG Stadiium", "Stadium"] = 'NRG Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Oakland Alameda-County Coliseum", "Stadium"] = 'Oakland Alameda County Coliseum'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Oakland-Alameda County Coliseum", "Stadium"] = 'Oakland Alameda County Coliseum'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Raymon James Stadium", "Stadium"] = 'Raymond James Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Solidier Field", "Stadium"] = 'Solider Field'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "Twickenham", "Stadium"] = 'Twickenham Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "University of Phoenix", "Stadium"] = 'University of Phoenix Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium"] == "US Bank Stadium", "Stadium"] = 'U.S. Bank Stadium'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outdoors", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "outdoor", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outdoors", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outdor", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outddors", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Oudoor", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Ourdoor", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outdoors ", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Outside", "Stadium_Type"] = 'Outdoor'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Indoor", "Stadium_Type"] = 'Indoors'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Indoor, fixed roof", "Stadium_Type"] = 'Indoor, Fixed Roof'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Dome, closed", "Stadium_Type"] = 'Closed Dome'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Domed, Closed", "Stadium_Type"] = 'Closed Dome'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Retr. Roof-Closed", "Stadium_Type"] = 'Retr. Roof - Closed'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Retr. roof - closed", "Stadium_Type"] = 'Retr. Roof - Closed'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Retr. Roof Closed", "Stadium_Type"] = 'Retr. Roof - Closed'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Indoor, non-retractable roof", "Stadium_Type"] = 'Indoor, Non-Retractable Roof'
Game_Data_Set.loc[Game_Data_Set["Stadium_Type"] == "Retr. Roof-Open", "Stadium_Type"] = 'Retr. Roof - Open'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "FieldTurf", "Turf"] = 'Field Turf'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "FieldTurf360", "Turf"] = 'Field Turf 360'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "FieldTurf 360", "Turf"] = 'Field Turf 360'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "grass", "Turf"] = 'Grass'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "Natrual Grass", "Turf"] = 'Natural Grass'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "Natural grass", "Turf"] = 'Natural Grass'
Game_Data_Set.loc[Game_Data_Set["Turf"] == "Naturall Grass", "Turf"] = 'Natural Grass'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Mostly cloudy", "Game_Weather"] = 'Mostly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Mostly Coudy", "Game_Weather"] = 'Mostly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Mostly CLoudy", "Game_Weather"] = 'Mostly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Partly sunny", "Game_Weather"] = 'Partly Sunny'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "cloudy", "Game_Weather"] = 'Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Party Cloudy", "Game_Weather"] = 'Partly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Partly CLoudy", "Game_Weather"] = 'Partly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Partly cloudy", "Game_Weather"] = 'Partly Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Suny", "Game_Weather"] = 'Sunny'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Sunny intervals", "Game_Weather"] = 'Sunny Intervals'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Snow showers", "Game_Weather"] = 'Snow Showers'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Indoor", "Game_Weather"] = 'Indoors'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Coudy", "Game_Weather"] = 'Cloudy'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "CLEAR", "Game_Weather"] = 'Clear'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Cloudy and cold", "Game_Weather"] = 'Cloudy and Cold'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Controlled", "Game_Weather"] = 'Controlled Climate'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Sunny and cool", "Game_Weather"] = 'Sunny and Cool'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Sunny and warm", "Game_Weather"] = 'Sunny and Warm'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Sunny intervals", "Game_Weather"] = 'Sunny Intervals'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Clear and warm", "Game_Weather"] = 'Clear and Warm'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Clear skies", "Game_Weather"] = 'Clear Skies'
Game_Data_Set.loc[Game_Data_Set["Game_Weather"] == "Mostly Clear. Gusting ot 14.", "Game_Weather"] = 'Mostly Clear'
Game_Data_Set.loc[Game_Data_Set["Season_Type"] == "Reg", "Season_Type"] = 'Regular Season'
Game_Data_Set.loc[Game_Data_Set["Season_Type"] == "Pre", "Season_Type"] = 'Pre Season'
Game_Data_Set.loc[Game_Data_Set["Season_Type"] == "Post", "Season_Type"] = 'Post Season'
Play_Information_Data_Set.loc[Play_Information_Data_Set["Season_Type"] == "Reg", "Season_Type"] = 'Regular Season'
Play_Information_Data_Set.loc[Play_Information_Data_Set["Season_Type"] == "Pre", "Season_Type"] = 'Pre Season'
Play_Information_Data_Set.loc[Play_Information_Data_Set["Season_Type"] == "Post", "Season_Type"] = 'Post Season'
Video_Review_Data_Set.loc[Video_Review_Data_Set["Primary_Impact_Type"] == "Helmet-to-body", "Primary_Impact_Type"] = 'Helmet-to-Body'
Video_Review_Data_Set.loc[Video_Review_Data_Set["Primary_Impact_Type"] == "Helmet-to-ground", "Primary_Impact_Type"] = 'Helmet-to-Ground'
Video_Review_Data_Set.loc[Video_Review_Data_Set["Primary_Impact_Type"] == "Helmet-to-helmet", "Primary_Impact_Type"] = 'Helmet-to-Helmet'
f, axarr = plt.subplots(2, 2, figsize=(20, 10))

f.subplots_adjust(hspace=0.5)

sns.countplot(Game_Data_Set['Season_Year'], ax=axarr[0][0], color='#EC7063', order = Game_Data_Set['Season_Year'].value_counts().index)
axarr[0][0].set_title("Season Year", fontsize=14)

sns.countplot(Game_Data_Set['Season_Type'], ax=axarr[0][1], color='#9B59B6',order = Game_Data_Set['Season_Type'].value_counts().index)
axarr[0][1].set_title("Season Type", fontsize=14)

sns.countplot(Game_Data_Set['Week'], ax=axarr[1][0], color='#45B39D',order = Game_Data_Set['Week'].value_counts().index)
axarr[1][0].set_title("Week", fontsize=14)

sns.countplot(Game_Data_Set['Game_Day'], ax=axarr[1][1], color='#F39C12',order = Game_Data_Set['Game_Day'].value_counts().index)
axarr[1][1].set_title("Game Day", fontsize=14)

f, axarr = plt.subplots(2, 2, figsize=(20, 10))

f.subplots_adjust(hspace=0.5)

sns.countplot(Play_Information_Data_Set['Season_Year'], ax=axarr[0][0], color='#EC7063', order = Play_Information_Data_Set['Season_Year'].value_counts().index)
axarr[0][0].set_title("Season Year", fontsize=14)

sns.countplot(Play_Information_Data_Set['Season_Type'], ax=axarr[0][1], color='#45B39D',order = Play_Information_Data_Set['Season_Type'].value_counts().index)
axarr[0][1].set_title("Season Type", fontsize=14)

sns.countplot(Play_Information_Data_Set['Week'], ax=axarr[1][0], color='#9B59B6',order = Play_Information_Data_Set['Week'].value_counts().index)
axarr[1][0].set_title("Week", fontsize=14)

sns.countplot(Play_Information_Data_Set['Quarter'], ax=axarr[1][1], color='#F39C12',order = Play_Information_Data_Set['Quarter'].value_counts().index)
axarr[1][1].set_title("Quarter", fontsize=14)
f, axarr = plt.subplots(2, 2, figsize=(20, 10))

f.subplots_adjust(hspace=0.5)

sns.countplot(Video_Review_Data_Set['Season_Year'], ax=axarr[0][0], color='#EC7063', order = Video_Review_Data_Set['Season_Year'].value_counts().index)
axarr[0][0].set_title("Season Year", fontsize=14)

sns.countplot(Video_Review_Data_Set['Player_Activity_Derived'], ax=axarr[0][1], color='#45B39D',order = Video_Review_Data_Set['Player_Activity_Derived'].value_counts().index)
axarr[0][1].set_title("Player Activity Derived", fontsize=14)

sns.countplot(Video_Review_Data_Set['Primary_Impact_Type'], ax=axarr[1][0], color='#9B59B6',order = Video_Review_Data_Set['Primary_Impact_Type'].value_counts().index)
axarr[1][0].set_title("Primary Impact Type", fontsize=14)

sns.countplot(Video_Review_Data_Set['Primary_Partner_Activity_Derived'], ax=axarr[1][1], color='#F39C12',order = Video_Review_Data_Set['Primary_Partner_Activity_Derived'].value_counts().index)
axarr[1][1].set_title("Primary Partner Activity Derived", fontsize=14)
TotalGame2017 = Game_Data_Set[Game_Data_Set['Season_Year']==2017]['Season_Year'].value_counts()
TotalGame2017Percentage = round(TotalGame2017 / len(Game_Data_Set.Season_Year) * 100,2)

TotalGame2016 = Game_Data_Set[Game_Data_Set['Season_Year']==2016]['Season_Year'].value_counts()
TotalGame2016Percentage = round(TotalGame2016 / len(Game_Data_Set.Season_Year) * 100,2)

TotalPercentage = round(len(Game_Data_Set.Season_Year) / len(Game_Data_Set.Season_Year) * 100,2)

Field_1 = pd.Series({'Description': 'Year 2017',
                        'Total Records': int(TotalGame2017.values),
                         'Percentage' : float(TotalGame2017Percentage.values),
                    })
Field_2 = pd.Series({'Description': 'Year 2016',
                        'Total Records': int(TotalGame2016.values),
                         'Percentage' : float(TotalGame2016Percentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Total',
                        'Total Records': Game_Data_Set['Season_Year'].count(),
                         'Percentage' : TotalPercentage})
YearSummary = pd.DataFrame([Field_1,Field_2,Field_3], index=['1','2','3'])
YearSummary
labels = (np.array(Game_Data_Set["Season_Year"].unique()))
values = Game_Data_Set["Season_Year"].value_counts()
colors = ['#F15854 ', '#60BD68  ']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='percent+label', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#FFFFFF', width=2)))
py.offline.iplot([trace], filename='styled_pie_chart')
TotalSunday = Game_Data_Set[Game_Data_Set['Game_Day']=='Sunday']['Game_Day'].value_counts()
TotalSundayPercentage = round(TotalSunday / len(Game_Data_Set.Game_Day) * 100,2)

TotalThursday = Game_Data_Set[Game_Data_Set['Game_Day']=='Thursday']['Game_Day'].value_counts()
TotalThursdayPercentage = round(TotalThursday / len(Game_Data_Set.Game_Day) * 100,2)

TotalSaturday = Game_Data_Set[Game_Data_Set['Game_Day']=='Saturday']['Game_Day'].value_counts()
TotalSaturdayPercentage = round(TotalSaturday / len(Game_Data_Set.Game_Day) * 100,2)

TotalMonday = Game_Data_Set[Game_Data_Set['Game_Day']=='Monday']['Game_Day'].value_counts()
TotalMondayPercentage = round(TotalMonday / len(Game_Data_Set.Game_Day) * 100,2)

TotalFriday = Game_Data_Set[Game_Data_Set['Game_Day']=='Friday']['Game_Day'].value_counts()
TotalFridayPercentage = round(TotalFriday / len(Game_Data_Set.Game_Day) * 100,2)

TotalWednesday = Game_Data_Set[Game_Data_Set['Game_Day']=='Wednesday']['Game_Day'].value_counts()
TotalWednesdayPercentage = round(TotalWednesday / len(Game_Data_Set.Game_Day) * 100,2)

TotalPercentage = round(len(Game_Data_Set.Game_Day) / len(Game_Data_Set.Game_Day) * 100,2)

Field_1 = pd.Series({'Description': 'Sunday',
                        'Total Records': int(TotalSunday.values),
                         'Percentage' : float(TotalSundayPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Thursday',
                        'Total Records': int(TotalThursday.values),
                         'Percentage' : float(TotalThursdayPercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Saturday',
                        'Total Records': int(TotalSaturday.values),
                         'Percentage' : float(TotalSaturdayPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Monday',
                        'Total Records': int(TotalMonday.values),
                         'Percentage' : float(TotalMondayPercentage.values),                     
                    })
Field_5 = pd.Series({'Description': 'Friday',
                        'Total Records': int(TotalFriday.values),
                         'Percentage' : float(TotalFridayPercentage.values),                     
                    })
Field_6 = pd.Series({'Description': 'Wednesday',
                        'Total Records': int(TotalWednesday.values),
                         'Percentage' : float(TotalWednesdayPercentage.values),                     
                    })
Field_7 = pd.Series({'Description': 'Total',
                        'Total Records': Game_Data_Set['Game_Day'].count(),
                         'Percentage' : TotalPercentage})
GameDaySummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5,Field_6,Field_7], index=['1','2','3','4','5','6','7'])
GameDaySummary
GraphData=Game_Data_Set.groupby('Game_Day').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Total Records', title='Game Day')
TotalReg = Game_Data_Set[Game_Data_Set['Season_Type']=='Regular Season']['Season_Type'].value_counts()
TotalRegPercentage = round(TotalReg / len(Game_Data_Set.Season_Type) * 100,2)

TotalPre = Game_Data_Set[Game_Data_Set['Season_Type']=='Pre Season']['Season_Type'].value_counts()
TotalPrePercentage = round(TotalPre / len(Game_Data_Set.Season_Type) * 100,2)

TotalPost = Game_Data_Set[Game_Data_Set['Season_Type']=='Post Season']['Season_Type'].value_counts()
TotalPostPercentage = round(TotalPost / len(Game_Data_Set.Season_Type) * 100,2)

TotalPercentage = round(len(Game_Data_Set.Season_Type) / len(Game_Data_Set.Season_Type) * 100,2)

Field_1 = pd.Series({'Description': 'Regular Season',
                        'Total Records': int(TotalReg.values),
                         'Percentage' : float(TotalRegPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Pre Season',
                        'Total Records': int(TotalPre.values),
                         'Percentage' : float(TotalPrePercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Post Season',
                        'Total Records': int(TotalPost.values),
                         'Percentage' : float(TotalPostPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Total',
                        'Total Records': Game_Data_Set['Season_Type'].count(),
                         'Percentage' : TotalPercentage})
SeasonTypeSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4], index=['1','2','3','4'])
SeasonTypeSummary
GraphData=Game_Data_Set.groupby('Season_Type').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Total Records', title='Season Type')
TotalGame2017 = Play_Information_Data_Set[Play_Information_Data_Set['Season_Year']==2017]['Season_Year'].value_counts()
TotalGame2017Percentage = round(TotalGame2017 / len(Play_Information_Data_Set.Season_Year) * 100,2)

TotalGame2016 = Play_Information_Data_Set[Play_Information_Data_Set['Season_Year']==2016]['Season_Year'].value_counts()
TotalGame2016Percentage = round(TotalGame2016 / len(Play_Information_Data_Set.Season_Year) * 100,2)

TotalPercentage = round(len(Play_Information_Data_Set.Season_Year) / len(Play_Information_Data_Set.Season_Year) * 100,2)

Field_1 = pd.Series({'Description': 'Year 2017',
                        'Total Records': int(TotalGame2017.values),
                         'Percentage' : float(TotalGame2017Percentage.values),
                    })
Field_2 = pd.Series({'Description': 'Year 2016',
                        'Total Records': int(TotalGame2016.values),
                         'Percentage' : float(TotalGame2016Percentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Total',
                        'Total Records': Play_Information_Data_Set['Season_Year'].count(),
                         'Percentage' : TotalPercentage})
YearSummary = pd.DataFrame([Field_1,Field_2,Field_3], index=['1','2','3'])
YearSummary
labels = (np.array(Play_Information_Data_Set["Season_Year"].unique()))
values = Play_Information_Data_Set["Season_Year"].value_counts()
colors = ['#F15854', '#60BD68']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='percent+label', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#FFFFFF', width=2)))
py.offline.iplot([trace], filename='styled_pie_chart')
TotalReg = Play_Information_Data_Set[Play_Information_Data_Set['Season_Type']=='Regular Season']['Season_Type'].value_counts()
TotalRegPercentage = round(TotalReg / len(Play_Information_Data_Set.Season_Type) * 100,2)

TotalPre = Play_Information_Data_Set[Play_Information_Data_Set['Season_Type']=='Pre Season']['Season_Type'].value_counts()
TotalPrePercentage = round(TotalPre / len(Play_Information_Data_Set.Season_Type) * 100,2)

TotalPost = Play_Information_Data_Set[Play_Information_Data_Set['Season_Type']=='Post Season']['Season_Type'].value_counts()
TotalPostPercentage = round(TotalPost / len(Play_Information_Data_Set.Season_Type) * 100,2)

TotalPercentage = round(len(Play_Information_Data_Set.Season_Type) / len(Play_Information_Data_Set.Season_Type) * 100,2)

Field_1 = pd.Series({'Description': 'Regular Season',
                        'Total Records': int(TotalReg.values),
                         'Percentage' : float(TotalRegPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Pre Season',
                        'Total Records': int(TotalPre.values),
                         'Percentage' : float(TotalPrePercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Post Season',
                        'Total Records': int(TotalPost.values),
                         'Percentage' : float(TotalPostPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Total',
                        'Total Records': Play_Information_Data_Set['Season_Type'].count(),
                         'Percentage' : TotalPercentage})
SeasonTypeSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4], index=['1','2','3','4'])
SeasonTypeSummary
GraphData=Play_Information_Data_Set.groupby('Season_Type').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Total Records', title='Season Type')
TotalSecond = Play_Information_Data_Set[Play_Information_Data_Set['Quarter']==2]['Quarter'].value_counts()
TotalSecondPercentage = round(TotalSecond / len(Play_Information_Data_Set.Quarter) * 100,2)

TotalFirst = Play_Information_Data_Set[Play_Information_Data_Set['Quarter']==1]['Quarter'].value_counts()
TotalFirstPercentage = round(TotalFirst / len(Play_Information_Data_Set.Quarter) * 100,2)

TotalFourth = Play_Information_Data_Set[Play_Information_Data_Set['Quarter']==4]['Quarter'].value_counts()
TotalFourthPercentage = round(TotalFourth / len(Play_Information_Data_Set.Quarter) * 100,2)

TotalThird = Play_Information_Data_Set[Play_Information_Data_Set['Quarter']==3]['Quarter'].value_counts()
TotalThirdPercentage = round(TotalThird / len(Play_Information_Data_Set.Quarter) * 100,2)

TotalOverTime = Play_Information_Data_Set[Play_Information_Data_Set['Quarter']==5]['Quarter'].value_counts()
TotalOverTimePercentage = round(TotalOverTime / len(Play_Information_Data_Set.Quarter) * 100,2)

TotalPercentage = round(len(Play_Information_Data_Set.Quarter) / len(Play_Information_Data_Set.Quarter) * 100,2)

Field_1 = pd.Series({'Description': 'Second Quarter',
                        'Total Records': int(TotalSecond.values),
                         'Percentage' : float(TotalSecondPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'First Quarter',
                        'Total Records': int(TotalFirst.values),
                         'Percentage' : float(TotalFirstPercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Fourth Quarter',
                        'Total Records': int(TotalFourth.values),
                         'Percentage' : float(TotalFourthPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Third Quarter',
                        'Total Records': int(TotalThird.values),
                         'Percentage' : float(TotalThirdPercentage.values),                     
                    })

Field_5 = pd.Series({'Description': 'Overtime',
                        'Total Records': int(TotalOverTime.values),
                         'Percentage' : float(TotalOverTimePercentage.values),                     
                    })

Field_6 = pd.Series({'Description': 'Total',
                        'Total Records': Play_Information_Data_Set['Quarter'].count(),
                         'Percentage' : TotalPercentage})
QuarterSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5,Field_6], index=['1','2','3','4','5','6'])
QuarterSummary
GraphData=Play_Information_Data_Set.groupby('Quarter').size().nlargest(6)
GraphData.iplot(kind='bar',yTitle='Total Records', title='Quarter')
TotalGame2017 = Video_Review_Data_Set[Video_Review_Data_Set['Season_Year']==2017]['Season_Year'].value_counts()
TotalGame2017Percentage = round(TotalGame2017 / len(Video_Review_Data_Set.Season_Year) * 100,2)

TotalGame2016 = Video_Review_Data_Set[Video_Review_Data_Set['Season_Year']==2016]['Season_Year'].value_counts()
TotalGame2016Percentage = round(TotalGame2016 / len(Video_Review_Data_Set.Season_Year) * 100,2)

TotalPercentage = round(len(Video_Review_Data_Set.Season_Year) / len(Video_Review_Data_Set.Season_Year) * 100,2)

Field_1 = pd.Series({'Description': 'Year 2017',
                        'Total Records': int(TotalGame2017.values),
                         'Percentage' : float(TotalGame2017Percentage.values),
                    })
Field_2 = pd.Series({'Description': 'Year 2016',
                        'Total Records': int(TotalGame2016.values),
                         'Percentage' : float(TotalGame2016Percentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Total',
                        'Total Records': Video_Review_Data_Set['Season_Year'].count(),
                         'Percentage' : TotalPercentage})
YearSummary = pd.DataFrame([Field_1,Field_2,Field_3], index=['1','2','3'])
YearSummary
labels = (np.array(Video_Review_Data_Set["Season_Year"].unique()))
values = Video_Review_Data_Set["Season_Year"].value_counts()
colors = ['#F15854', '#60BD68']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='percent+label', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#FFFFFF', width=2)))
py.offline.iplot([trace], filename='styled_pie_chart')
TotalTackling = Video_Review_Data_Set[Video_Review_Data_Set['Player_Activity_Derived']=='Tackling']['Player_Activity_Derived'].value_counts()
TotalTacklingPercentage = round(TotalTackling / len(Video_Review_Data_Set.Season_Year) * 100,2)

TotalBlocked = Video_Review_Data_Set[Video_Review_Data_Set['Player_Activity_Derived']=='Blocked']['Player_Activity_Derived'].value_counts()
TotalBlockedPercentage = round(TotalBlocked / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalBlocking = Video_Review_Data_Set[Video_Review_Data_Set['Player_Activity_Derived']=='Blocking']['Player_Activity_Derived'].value_counts()
TotalBlockingPercentage = round(TotalBlocking / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalTackled = Video_Review_Data_Set[Video_Review_Data_Set['Player_Activity_Derived']=='Tackled']['Player_Activity_Derived'].value_counts()
TotalTackledPercentage = round(TotalTackled / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalActivity = round(len(Video_Review_Data_Set.Player_Activity_Derived) / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

Field_1 = pd.Series({'Description': 'Tackling',
                        'Total Records': int(TotalTackling.values),
                         'Percentage' : float(TotalTacklingPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Blocked',
                        'Total Records': int(TotalBlocked.values),
                         'Percentage' : float(TotalBlockedPercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Blocking',
                        'Total Records': int(TotalBlocking.values),
                         'Percentage' : float(TotalBlockingPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Tackled',
                        'Total Records': int(TotalTackled.values),
                         'Percentage' : float(TotalTackledPercentage.values),                     
                    })
Field_5 = pd.Series({'Description': 'Total',
                        'Total Records': Video_Review_Data_Set['Player_Activity_Derived'].count(),
                         'Percentage' : TotalActivity})
ActivitySummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5], index=['1','2','3','4','5'])
ActivitySummary
GraphData=Video_Review_Data_Set.groupby('Player_Activity_Derived').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Number of Injuries', title='Player Activity Derived')
TotalHTB = Video_Review_Data_Set[Video_Review_Data_Set['Primary_Impact_Type']=='Helmet-to-Body']['Primary_Impact_Type'].value_counts()
TotalHTBPercentage = round(TotalHTB / len(Video_Review_Data_Set.Season_Year) * 100,2)

TotalHTH = Video_Review_Data_Set[Video_Review_Data_Set['Primary_Impact_Type']=='Helmet-to-Helmet']['Primary_Impact_Type'].value_counts()
TotalHTHPercentage = round(TotalHTH / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalHTG = Video_Review_Data_Set[Video_Review_Data_Set['Primary_Impact_Type']=='Helmet-to-Ground']['Primary_Impact_Type'].value_counts()
TotalHTGPercentage = round(TotalHTG / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalUnclear = Video_Review_Data_Set[Video_Review_Data_Set['Primary_Impact_Type']=='Unclear']['Primary_Impact_Type'].value_counts()
TotalUnclearPercentage = round(TotalUnclear / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

TotalActivity = round(len(Video_Review_Data_Set.Player_Activity_Derived) / len(Video_Review_Data_Set.Player_Activity_Derived) * 100,2)

Field_1 = pd.Series({'Description': 'Helmet-to-Body',
                        'Total Records': int(TotalHTB.values),
                         'Percentage' : float(TotalHTBPercentage.values),
                    })
Field_2 = pd.Series({'Description': 'Helmet-to-Helmet',
                        'Total Records': int(TotalHTH.values),
                         'Percentage' : float(TotalHTHPercentage.values),                     
                    })
Field_3 = pd.Series({'Description': 'Helmet-to-Ground',
                        'Total Records': int(TotalHTG.values),
                         'Percentage' : float(TotalHTGPercentage.values),                     
                    })
Field_4 = pd.Series({'Description': 'Unclear',
                        'Total Records': int(TotalUnclear.values),
                         'Percentage' : float(TotalUnclearPercentage.values),                     
                    })
Field_5 = pd.Series({'Description': 'Total',
                        'Total Records': Video_Review_Data_Set['Primary_Impact_Type'].count(),
                         'Percentage' : TotalActivity})
ImpactSummary = pd.DataFrame([Field_1,Field_2,Field_3,Field_4,Field_5], index=['1','2','3','4','5'])
ImpactSummary
GraphData=Video_Review_Data_Set.groupby('Primary_Impact_Type').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Number of Injuries', title='Primary Impact Type')
Game_Data_Set.Start_Time.value_counts()
GraphData=Game_Data_Set.groupby('Start_Time').size().nlargest(666)
GraphData.iplot(kind='bar',yTitle='Number of Match', title='Game Start Time')
Game_Video_Data_Set = pd.merge(Game_Data_Set, Video_Review_Data_Set,
                          how='inner',
                          on=['Game_Key'])
GraphData=Game_Video_Data_Set.groupby('Start_Time').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Number of Injuires', title='Game Injuries Start Time')
plt.figure(figsize = (18,5))
sns.swarmplot(x= Video_Review_Data_Set["Primary_Impact_Type"], y = Video_Review_Data_Set["Game_Key"])
plt.title("Total Primary Impact Type")
plt.xlabel("Primary Impact Type")
plt.ylabel("Total")
Game_Data_Set.Stadium.value_counts()
GraphData=Game_Data_Set.groupby('Stadium').size().nlargest(666)
GraphData.iplot(kind='bar',yTitle='Number of Match', title='Stadium Name')
Game_Video_Stadium_Data_Set = pd.merge(Game_Data_Set,Video_Review_Data_Set,
                          how='inner',
                          on=['Game_Key'])
Game_Video_Stadium_Data_Set.Stadium.value_counts()
GraphData=Game_Video_Stadium_Data_Set.groupby('Stadium').size().nlargest(10)
GraphData.iplot(kind='bar',yTitle='Number of Match Injuries', title='Stadium Name')
Game_Video_Data_Set = pd.merge(Video_Review_Data_Set,Play_Player_Role_Data_Set,
                          how='inner',
                          on=['Play_ID','Game_Key','GSISID'])
HTH = Game_Video_Data_Set[(Game_Video_Data_Set.Primary_Impact_Type == 'Helmet-to-Helmet')]
HTH.Role.value_counts()
GraphData=HTH.groupby('Role').size().nlargest(11)
GraphData.iplot(kind='bar',yTitle='Number of Injuires', title='Helmet-to-Helmet')
HTB = Game_Video_Data_Set[(Game_Video_Data_Set.Primary_Impact_Type == 'Helmet-to-Body')]
HTB.Role.value_counts()
GraphData=HTB.groupby('Role').size().nlargest(11)
GraphData.iplot(kind='bar',yTitle='Number of Injuires', title='Helmet-to-Body')