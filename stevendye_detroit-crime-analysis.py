import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style
style.use('ggplot')
df = pd.read_csv('../input/DPD__All_Crime_Incidents__January_1__2009_-_December_6__2016.csv')
df['INCIDENTDATE'] = pd.to_datetime(df.INCIDENTDATE)
df.NEIGHBORHOOD.fillna('Unknown', inplace=True)
df['YEAR']=df.INCIDENTDATE.dt.year

# Relabel Districts
df['COUNCIL'].replace('City Council District 5 - Belle Isle','City Council District 5', inplace=True)

# Merge CATEGORIES
df['CATEGORY'].replace('KIDNAPING','KIDNAPPING', inplace=True)
df['CATEGORY'].replace('TRAFFIC OFFENSES','TRAFFIC', inplace=True)
df['CATEGORY'].replace('JUSTIFIABLE HOMICIDE','HOMICIDE', inplace=True)
df['CATEGORY'].replace('OTHER BURGLARY','BURGLARY', inplace=True)
df['CATEGORY'].replace('MISCELLANEOUS ARREST','MISCELLANEOUS', inplace=True)

# This section moves automobile crimes out of the 'STOLEN PROPERTY' CATEGORY and into 'STOLEN VEHICLE'
df['OFFENSEDESCRIPTION'].replace('MOTOR VEHICLE AS STOLEN PROPERTY - THEFT AND STRIP (INCLUDES CHOP SHOP)',
                                 'MOTOR VEHICLE - THEFT AND STRIP', inplace=True)
df['OFFENSEDESCRIPTION'].replace('MOTOR VEHICLE AS STOLEN PROPERTY - THEFT AND SALE',
                                 'MOTOR VEHICLE - THEFT AND SALE', inplace=True) 
df['OFFENSEDESCRIPTION'].replace('MOTOR VEHICLE AS STOLEN PROPERTY - UNAUTHORIZED USE (INCLUDES JOY RIDING)',
                                 'MOTOR VEHICLE - UNAUTHORIZED USE', inplace=True) 
df['OFFENSEDESCRIPTION'].replace('MOTOR VEHICLE AS STOLEN PROPERTY (OTHER)','STOLEN (OTHER)', inplace=True) 
df.loc[df['OFFENSEDESCRIPTION'] == 'MOTOR VEHICLE - THEFT AND STRIP', 'CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] == 'MOTOR VEHICLE - THEFT AND SALE', 'CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] == 'MOTOR VEHICLE - UNAUTHORIZED USE', 'CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] == 'STOLEN (OTHER)', 'CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] == 'POSSESS/RECEIVE STOLEN VEHICLE/PARTS','CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] == 'MOTOR VEHICLE AS STOLEN PROPERTY - THEFT AND USE','CATEGORY'] = 'STOLEN VEHICLE'
df.loc[df['OFFENSEDESCRIPTION'] ==  'INTERSTATE TRANSPORT OF STOLEN VEHICLE', 'CATEGORY'] = 'STOLEN VEHICLE'
df_2016 = df[df.YEAR==2016]
D1N = df_2016[df_2016.COUNCIL=='City Council District 1'].NEIGHBORHOOD.unique()
D2N = df_2016[df_2016.COUNCIL=='City Council District 2'].NEIGHBORHOOD.unique()
D3N = df_2016[df_2016.COUNCIL=='City Council District 3'].NEIGHBORHOOD.unique()
D4N = df_2016[df_2016.COUNCIL=='City Council District 4'].NEIGHBORHOOD.unique()
D5N = df_2016[df_2016.COUNCIL=='City Council District 5'].NEIGHBORHOOD.unique()
D6N = df_2016[df_2016.COUNCIL=='City Council District 6'].NEIGHBORHOOD.unique()
D7N = df_2016[df_2016.COUNCIL=='City Council District 7'].NEIGHBORHOOD.unique()
df.head()
plt.plot(df.YEAR.value_counts().sort_index())
plt.title('Crime per Year')
plt.ylabel('Crime Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,200000));
df['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2009 = df[df.YEAR==2009]
df_2009['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2010 = df[df.YEAR==2010]
df_2010['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2011 = df[df.YEAR==2011]
df_2011['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2012 = df[df.YEAR==2012]
df_2012['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2013 = df[df.YEAR==2013]
df_2013['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2014 = df[df.YEAR==2014]
df_2014['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2015 = df[df.YEAR==2015]
df_2015['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_2016['CATEGORY'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Type of Crime', color='blue')
df_D1 = df.loc[df['NEIGHBORHOOD'].isin(D1N)]
df_D1['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 1', color='yellow')
df_D2 = df.loc[df['NEIGHBORHOOD'].isin(D2N)]
df_D2['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 2', color='lime')
df_D3 = df.loc[df['NEIGHBORHOOD'].isin(D3N)]
df_D3['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 3', color='red')
df_D4 = df.loc[df['NEIGHBORHOOD'].isin(D4N)]
df_D4['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 4', color='purple')
df_D5 = df.loc[df['NEIGHBORHOOD'].isin(D5N)]
df_D5['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 5', color='pink')
df_D6 = df.loc[df['NEIGHBORHOOD'].isin(D6N)]
df_D6['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 6', color='gold')
df_D7 = df.loc[df['NEIGHBORHOOD'].isin(D7N)]
df_D7['NEIGHBORHOOD'].value_counts().sort_index().plot.bar(figsize=(20, 6),
        title='Crime per Neighborhood: District 7', color='lightblue')
df[df.CATEGORY=='ABORTION'].drop(columns=['CATEGORY','YEAR','COUNCIL'])
df_Agg_Assault = df[df.CATEGORY=='AGGRAVATED ASSAULT']
plt.plot(df_Agg_Assault.YEAR.value_counts().sort_index())
plt.title('Aggravated Assaults per Year')
plt.ylabel('Aggravated Assault Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,20000));
df_Arson = df[df.CATEGORY=='ARSON']
plt.plot(df_Arson.YEAR.value_counts().sort_index())
plt.title('Arsons per Year')
plt.ylabel('Arson Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,1500));
df_Assault = df[df.CATEGORY=='ASSAULT']
plt.plot(df_Assault.YEAR.value_counts().sort_index())
plt.title('Assaults per Year')
plt.ylabel('Assault Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,30000));
df_Bribery = df[df.CATEGORY=='BRIBERY']
plt.plot(df_Bribery.YEAR.value_counts().sort_index())
plt.title('Bribery Crimes per Year')
plt.ylabel('Bribery Crimes Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_Burglary = df[df.CATEGORY=='BURGLARY']
plt.plot(df_Burglary.YEAR.value_counts().sort_index())
plt.title('Burglary per Year')
plt.ylabel('Burglary Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,25000));
df_Civil = df[df.CATEGORY=='CIVIL']
df_Civil_D = df_Civil[df_Civil.OFFENSEDESCRIPTION =='CIVIL CUSTODIES - DIVORCE AND SUPPORT']
df_Civil_W = df_Civil[df_Civil.OFFENSEDESCRIPTION =='CIVIL CUSTODIES - WALK AWAY - MENTAL INSTITUTIONS AND HOSPITALS']
df_Civil_A = df_Civil[df_Civil.OFFENSEDESCRIPTION =='CIVIL INFRACTION - ILLEGAL POSSESSION OF ALCOHOLIC LIQUOR']
plt.plot(df_Civil.YEAR.value_counts().sort_index(),color='red')
plt.plot(df_Civil_D.YEAR.value_counts().sort_index(),color='blue')
plt.plot(df_Civil_W.YEAR.value_counts().sort_index(),color='green')
plt.plot(df_Civil_A.YEAR.value_counts().sort_index(),color='orange')
plt.title('Civil Crimes per Year')
plt.ylabel('Civil Crime Count')
plt.xlabel('Year')
total_patch = mpatches.Patch(color='red',label='Total')
divorce_patch = mpatches.Patch(color='blue',label='Divorce and Support')
walk_patch = mpatches.Patch(color='green',label='Walk Away')
alcohol_patch = mpatches.Patch(color='orange',label='Illegal Possession of Alcohol')
plt.legend(handles=[total_patch,divorce_patch,walk_patch,alcohol_patch])
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_DtP = df[df.CATEGORY=='DAMAGE TO PROPERTY']
plt.plot(df_DtP.YEAR.value_counts().sort_index())
plt.title('Damage to Property Crimes per Year')
plt.ylabel('Damage to Property Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,20000));
df_DD = df[df.CATEGORY=='DANGEROUS DRUGS']
plt.plot(df_DD.YEAR.value_counts().sort_index())
plt.title('Dangerous Drugs Crimes per Year')
plt.ylabel('Dangerous Drugs Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,6000));
df_DD.OFFENSEDESCRIPTION.unique()
df_DC = df[df.CATEGORY=='DISORDERLY CONDUCT']
plt.plot(df_DC.YEAR.value_counts().sort_index())
plt.title('Disorderly Conduct Crimes per Year')
plt.ylabel('Disorderly Conduct Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3000));
df_Drunkenness = df[df.CATEGORY=='DRUNKENNESS']
plt.plot(df_Drunkenness.YEAR.value_counts().sort_index())
plt.title('Drunkenness Crimes per Year')
plt.ylabel('Drunkenness Crimes Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_Embezzlement = df[df.CATEGORY=='EMBEZZLEMENT']
plt.plot(df_Embezzlement.YEAR.value_counts().sort_index())
plt.title('Embezzlements per Year')
plt.ylabel('Embezzlement Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_Environment = df[df.CATEGORY=='ENVIRONMENT']
plt.plot(df_Environment.YEAR.value_counts().sort_index())
plt.title('Environment per Year')
plt.ylabel('Environment Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,200));
df_Environment.OFFENSEDESCRIPTION.unique()
df_Escape = df[df.CATEGORY=='ESCAPE']
plt.plot(df_Escape.YEAR.value_counts().sort_index())
plt.title('Escapes per Year')
plt.ylabel('Escape Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3000));
df_Extortion = df[df.CATEGORY=='EXTORTION']
plt.plot(df_Extortion.YEAR.value_counts().sort_index())
plt.title('Extortions per Year')
plt.ylabel('Extortions Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,300));
df_FO = df[df.CATEGORY=='FAMILY OFFENSE']
df_FO.OFFENSEDESCRIPTION.unique()
plt.plot(df_FO.YEAR.value_counts().sort_index())
plt.title('Family Offense Crimes per Year')
plt.ylabel('Family Offense Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,600));
df_Fraud = df[df.CATEGORY=='FRAUD']
plt.plot(df_Fraud.YEAR.value_counts().sort_index())
plt.title('Frauds per Year')
plt.ylabel('Fraud Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,8000));
df_Forgery = df[df.CATEGORY=='FORGERY']
plt.plot(df_Forgery.YEAR.value_counts().sort_index())
plt.title('Forgery Crimes per Year')
plt.ylabel('Forgery Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,400));
df_Gambling = df[df.CATEGORY=='GAMBLING']
plt.plot(df_Gambling.YEAR.value_counts().sort_index())
plt.title('Gambling Crimes per Year')
plt.ylabel('Gambling Crime Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_Homicide = df[df.CATEGORY=='HOMICIDE']
df_Homicide_JH = df_Homicide[df_Homicide.OFFENSEDESCRIPTION =='JUSTIFIABLE HOMICIDE']
plt.plot(df_Homicide.YEAR.value_counts().sort_index(),color='red')
plt.plot(df_Homicide_JH.YEAR.value_counts().sort_index(),color='blue')
plt.title('Homicides per Year')
plt.ylabel('Homicide Count')
plt.xlabel('Year')
total_patch = mpatches.Patch(color='red',label='Total')
JH_patch = mpatches.Patch(color='blue',label='Justifiable Homicide')
plt.legend(handles=[total_patch,JH_patch])
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,500));
df_Homicide.OFFENSEDESCRIPTION.unique()
df_Immigration = df[df.CATEGORY=='IMMIGRATION']
plt.plot(df_Immigration.YEAR.value_counts().sort_index())
plt.title('Immigration Crimes per Year')
plt.ylabel('Immigaration Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_Kidnapping = df[df.CATEGORY=='KIDNAPPING']
plt.plot(df_Kidnapping.YEAR.value_counts().sort_index())
plt.title('Kidnapping Crimes per Year')
plt.ylabel('Kidnapping Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,500));
df_Larceny = df[df.CATEGORY=='LARCENY']
plt.plot(df_Larceny.YEAR.value_counts().sort_index())
plt.title('Larceny per Year')
plt.ylabel('Larceny Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,25000));
df_Liquor = df[df.CATEGORY=='LIQUOR']
plt.plot(df_Liquor.YEAR.value_counts().sort_index())
plt.title('Liquor Crime per Year')
plt.ylabel('Liquor Crime Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,200));
df_Military = df[df.CATEGORY=='MILITARY']
df_Military.drop(columns=['CATEGORY','YEAR','COUNCIL'])
df_Miscellaneous = df[df.CATEGORY=='MISCELLANEOUS']
df_Miscellaneous['OFFENSEDESCRIPTION'].unique()
plt.plot(df_Miscellaneous.YEAR.value_counts().sort_index())
plt.title('Miscellaneous Crimes per Year')
plt.ylabel('Miscellaneous Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,40000));
df_MI = df[df.CATEGORY=='MURDER/INFORMATION']
df_MI['OFFENSEDESCRIPTION'].unique()
plt.plot(df_MI.YEAR.value_counts().sort_index())
plt.title('Murder/Informations per Year')
plt.ylabel('Muder/Information Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,20000));
df_NH = df[df.CATEGORY=='NEGLIGENT HOMICIDE']
plt.plot(df_NH.YEAR.value_counts().sort_index())
plt.title('Negligent Homicide Crimes per Year')
plt.ylabel('Negligent Homicide Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_NH.OFFENSEDESCRIPTION.count()
df_NH.OFFENSEDESCRIPTION.unique()
df_Obscenity = df[df.CATEGORY=='OBSCENITY']
plt.plot(df_Obscenity.YEAR.value_counts().sort_index())
plt.title('Obscenity Judiciary Crimes per Year')
plt.ylabel('Obscenity Judiciary Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100));
df_OJ = df[df.CATEGORY=='OBSTRUCTING JUDICIARY']
plt.plot(df_OJ.YEAR.value_counts().sort_index())
plt.title('Obstructing Judiciary Crimes per Year')
plt.ylabel('Obstructing Judiciary Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3000));
df_OtP = df[df.CATEGORY=='OBSTRUCTING THE POLICE']
plt.plot(df_OtP.YEAR.value_counts().sort_index())
plt.title('Obstructing the Police Crimes per Year')
plt.ylabel('Obstructing the Police Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,600));
df_Other = df[df.CATEGORY=='OTHER']
df_Other['OFFENSEDESCRIPTION'].unique()
plt.plot(df_Other.YEAR.value_counts().sort_index())
plt.title('Other Crimes per Year')
plt.ylabel('Other Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,800));
df_OUIL = df[df.CATEGORY=='OUIL']
plt.plot(df_OUIL.YEAR.value_counts().sort_index())
plt.title('OUIL per Year')
plt.ylabel('OUIL Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3000));
df_Robbery = df[df.CATEGORY=='ROBBERY']
plt.plot(df_Robbery.YEAR.value_counts().sort_index())
plt.title('Robberies per Year')
plt.ylabel('Robbery Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,8000));
df_Runaway = df[df.CATEGORY=='RUNAWAY']
plt.plot(df_Runaway.YEAR.value_counts().sort_index())
plt.title('Runaway Crimes per Year')
plt.ylabel('Runaway Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,1000));
df_SO = df[df.CATEGORY=='SEX OFFENSES']
df_SO.drop(columns=['CATEGORY','YEAR','COUNCIL'])
df_Solicitation = df[df.CATEGORY=='SOLICITATION']
plt.plot(df_Solicitation.YEAR.value_counts().sort_index())
plt.title('Solicitation Crimes per Year')
plt.ylabel('Solicitation Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,800));
df_SP = df[df.CATEGORY=='STOLEN PROPERTY']
plt.plot(df_SP.YEAR.value_counts().sort_index())
plt.title('Stolen Property Crimes per Year')
plt.ylabel('Stolen Property Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,1000));
df_SV = df[df.CATEGORY=='STOLEN VEHICLE']
plt.plot(df_SV.YEAR.value_counts().sort_index())
plt.title('Stolen Vehicles per Year')
plt.ylabel('Stolen Vehice Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,18000));
df_Traffic = df[df.CATEGORY=='TRAFFIC']
plt.plot(df_Traffic.YEAR.value_counts().sort_index())
plt.title('Traffic Crimes per Year')
plt.ylabel('Traffic Crimes Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,14000));
df_Vagrancy = df[df.CATEGORY=='VAGRANCY (OTHER)']
plt.plot(df_Vagrancy.YEAR.value_counts().sort_index())
plt.title('Vagrancy Crimes per Year')
plt.ylabel('Vagrancy Crimes Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,400));
df_WO = df[df.CATEGORY=='WEAPONS OFFENSES']
plt.plot(df_WO.YEAR.value_counts().sort_index())
plt.title('Weapons Offenses per Year')
plt.ylabel('Weapons Offenses Count')
plt.xlabel('Year')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,3000));