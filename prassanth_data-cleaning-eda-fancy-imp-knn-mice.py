#Importing libraries

import re

import numpy as np

import pandas as pd

pd.pandas.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 250)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import warnings 

warnings.filterwarnings('ignore')
#Loading csv

df = pd.read_csv('../input/new-cars-price-2019/New_cars_price.csv')

df.head()
df.shape
#Number of missing values across columns



def missing_values(dataframe):

    null_counts = dataframe.isnull().sum()

    mean_missing = dataframe.isnull().mean()*100

    missing_val = pd.DataFrame({'Count' : null_counts[null_counts > 0] , 'Percentage Missing(%)' : mean_missing[mean_missing > 0] })

    missing_val.sort_values(by = 'Count' , inplace=True)

    missing_val.reset_index(inplace=True)

    missing_val.columns = ['Features' , 'Count' , 'Percentage Missing(%)' ]

    return missing_val

missing = missing_values(df)

missing
df1 = df.copy()
df1['Manufacturer'] = df1['Model'].str.split(' ').str[1]     #Manufacturing Company

df1['Model year'] = df1['Model'].str.split(' ').str[0]       #Model year

df1.drop(columns='Model' , inplace  = True)
df1['Manufacturer'].replace({'Alfa':'Alfa Romeo' , 

                             'Aston':'Aston Martin',

                             'FIAT':'Fiat',

                             'INFINITI':'Infiniti',

                             'Land':'Land Rover',

                             'MINI':'Mini',

                             'smart':'Smart'} , inplace = True)
sns.set_style('whitegrid')

plt.figure(figsize=(20,10))

g = sns.countplot(

    data=df1,

    x='Manufacturer',

    order = df1['Manufacturer'].value_counts().index,

    palette='PuBuGn_d'

)

g.set_title('Car Manufacturers')

plt.xticks(rotation=90)
df1['MSRP'] = df1['MSRP'].str.replace("$", "").str.replace(",", "").astype(float)
plt.figure(figsize=[20,10])

sns.set(font_scale=1.25)

sns.distplot(df1['MSRP'].dropna(),color='r')

plt.xlabel('Price')

plt.ylabel('PDF')
sns.set(font_scale=1.25)

plt.figure(figsize=(20,10))

g = sns.barplot(

    data=df1,

    x='Manufacturer',

    y='MSRP',

    order = df1.groupby(by = 'Manufacturer')['MSRP'].mean().sort_values( ascending=False).index ,

    palette='pastel'

)

g.set(ylabel = 'Average Price in dollars')

g.set_title('Car Manufacturers')

plt.xticks(rotation=90)
df1['Engine'] = df1['Engine'].str.split(',').str[0].str.split(' ').str[-1]



df1['Engine'].replace({'I-4':'l4','V-6':'V6','I4':'l4',

                       'V-12':'V12','V-8':'V8','I-5':'l5',

                       'I5':'l5','W-12':'W12','I-6':'l6',

                       '6-Cyl':'Flat','Cyl':'Flat','I-3':'l3',

                       'L4':'l4','Turbocharged':'l4','Gas':'l4',

                       '4-Cyl':'l4','5-Cyl':'l5','ECOTEC':'l4',

                       'Diesel':'l4','(Vortec)':'V8','I3':'l3',

                       'V-10':'V10','i4':'l4','4-cyl':'l4',

                       'H-6':'Flat','6':'Flat','6-cyl':'Flat',

                       'H-4':'Flat','4':'Flat','Electric/Gas':'Electric'}, inplace = True)
sns.set(font_scale=1.25)

plt.figure(figsize=(15,8))

g = sns.countplot(

    data=df1,

    y='Engine',

    order = df1['Engine'].value_counts().index,

    palette='deep'

)

g.set_title('Engine Configuration')

plt.figure(figsize=(20,10))

g = sns.barplot(

    data=df1,

    x='Engine',

    y='MSRP',

    order = df1.groupby(by = 'Engine')['MSRP'].mean().sort_values( ascending=False).index ,

    palette='muted'

)

g.set(ylabel = 'Average Price in dollars')

g.set_title('Car Manufacturers')
plt.figure(figsize=[15,8])

sns.boxplot(x='Engine' , y='MSRP',data=df1)

plt.tight_layout()

plt.title('Boxplot between Engine type and price')
df['Suspension Type - Front'].value_counts()
df1['Suspension Type - Front'] = df['Suspension Type - Front']

#Suspension type Front



pattern1 = re.compile(r'(?i)(strut|MacPh|Mcpher)')                          #Strut

pattern2 = re.compile(r'(?i)(bone|short|sla|pivot|upper)')                  #Wishbone

pattern3 = re.compile(r'(?i)(tors|twis|crank|torq)')                        #Torsion bar

pattern4 = re.compile(r'(?i)(coil)')                                        #Coil spring

pattern5 = re.compile(r'(?i)(link|trap|control arm|multi|Trailing arm)')    #Link type

pattern6 = re.compile(r'(?i)(crank|solid|axle)')                            #Axle

pattern7 = re.compile(r'(?i)(leaf|stage|hotch|hypo)')                       #Leaf type

pattern8 = re.compile(r'(?i)(air)')                                         #Air suspension

pattern9 = re.compile(r'(?i)(indep|indpen|indep)')                          #Independent



df1['Suspension Type - Front'].fillna('NA',inplace = True)

df1['Suspension Type - Rear'].fillna('NA',inplace = True)



def suspension(x):

    if re.search(pattern1,x):

        return 'MacPherson Strut'

    elif re.search(pattern2,x):

        return 'Double Wishbone'

    elif re.search(pattern3,x):

        return 'Torsion Bar'

    elif re.search(pattern4,x):

        return 'Coil Spring'

    elif re.search(pattern5,x):

        return 'Link type'

    elif re.search(pattern6,x):

        return 'Axle'

    elif re.search(pattern7,x):

        return 'Leaf type'

    elif re.search(pattern8,x):

        return 'Air Suspension'

    elif re.search(pattern9,x):

        return 'Independent'

    else:

        return 'Others'
df1['Suspension Type - Front'] = df1['Suspension Type - Front'].apply(suspension)

df1['Suspension Type - Front'].value_counts()
df1['Suspension Type - Rear'] = df1['Suspension Type - Rear'].apply(suspension)

df1['Suspension Type - Rear'].value_counts()
p = dict(zip(df1['Suspension Type - Front'].unique(), sns.color_palette()))

fig, ax =plt.subplots(2,1,figsize=(20,15))

sns.countplot(df1['Suspension Type - Front'],palette=p,order=df1['Suspension Type - Front'].value_counts().index ,ax=ax[0])

sns.countplot(df1['Suspension Type - Rear'],palette=p ,order=df1['Suspension Type - Rear'].value_counts().index ,ax=ax[1])
df1['EPA Fuel Economy Est - City (MPG)'] = df1['EPA Fuel Economy Est - City (MPG)'].str.split(' ').str[0].astype('float')
df1['Base Curb Weight (lbs)'] = df1['Base Curb Weight (lbs)'].str.replace(',','').str.split(' ').str[0].str.split('-').str[0].astype('float')
df1['Passenger Volume (ft³)'] = df1['Passenger Volume (ft³)'].replace('-TBD-',np.NAN).astype('float')
df1['Height, Overall (in)'] = df1['Height, Overall (in)'].str.split(' ').str[0].astype('float')
df1['Fuel Tank Capacity, Approx (gal)'] = df1['Fuel Tank Capacity, Approx (gal)'].str.split(' ').str[0].astype('float')
df1['Body Style'].replace('Crew Cab Pickup', 'Crew Cab Pickup - Standard Bed' , inplace = True)

df1['Body Style'].replace('Extended Cab Pickup', 'Extended Cab Pickup - Standard Bed' , inplace = True)

df1['Body Style'].replace('Regular Cab Chassis-Cab', 'Regular Cab Pickup - Standard Bed',inplace = True)

df1['Body Style'].replace('3dr Car' , 'Hatchback' , inplace = True)

df1['Body Style'].replace(['Crew Cab Pickup','Extended Cab Pickup'] , np.nan , inplace = True)





df1['Category'] = df1['Body Style']



van = ['Mini-van, Cargo', 'Full-size Passenger Van', 'Full-size Cargo Van', 'Mini-van, Passenger', 'Specialty Vehicle']



pickups = ['Crew Cab Pickup - Short Bed', 'Crew Cab Pickup - Standard Bed', 

        'Extended Cab Pickup - Short Bed', 'Extended Cab Pickup - Standard Bed',

        'Extended Cab Pickup - Long Bed', 'Regular Cab Pickup - Long Bed', 

        'Crew Cab Pickup - Long Bed', 'Regular Cab Pickup - Short Bed', 

        'Regular Cab Pickup - Standard Bed', 'Extended Cab Pickup', 'Crew Cab Pickup',

        'Regular Cab Chassis-Cab', 'Pickup - Short Bed', 'Pickup - Standard Bed', 'Pickup - Long Bed']



car = ['2dr Car', '4dr Car', 'Convertible', 'Station Wagon', '3dr Car', 'Hatchback']



df1['Category'] = df1['Category'].str.replace('Sport Utility', 'SUV')



for item in van:

    df1['Category'] = df1['Category'].str.replace(item, 'Van')

    

for item in pickups:

    df1['Category'] = df1['Category'].str.replace(item, 'Pickup')

    

for item in car:

    df1['Category'] = df1['Category'].str.replace(item, 'Car')
df1['Drivetrain'].replace(['Front Wheel Drive', 'Front-Wheel Drive' , 

                           'Front wheel drive','Front-wheel drive' ,

                           '2 Wheel Drive' , '2WD' , '2-Wheel Drive'] , 'FWD' , inplace = True)

df1['Drivetrain'].replace(['Rear Wheel Drive' , 'REAR WHEEL DRIVE' , 

                           'Rear-Wheel Drive' ,'Rear wheel drive'] , 'RWD' , inplace = True)

df1['Drivetrain'].replace(['All Wheel Drive' , 'All-Wheel Drive' ,

                           'All wheel drive' , 'All-wheel drive' ] , 'AWD' , inplace = True)

df1['Drivetrain'].replace(['4 Wheel Drive' , 'Four Wheel Drive' , 

                           '4-Wheel Drive' , 'Four-Wheel Drive' , 

                           '4-wheel Drive' , 'Four wheel drive'] , '4WD' , inplace = True)
fig, ax =plt.subplots(1,2,figsize=(18,8))

sns.barplot(df1['Drivetrain'] , df1['MSRP'] , ax=ax[0])

sns.countplot(df1['Category'] , hue=df1['Drivetrain'] , ax=ax[1])
g = sns.FacetGrid(df1 , col='Drivetrain')

g.map(sns.boxplot , 'MSRP',color='g')

g.fig.suptitle('Boxplot between Price and Drivetrain types',y=1.06)
df1['Trans Type'].replace('10.0' , '10' , inplace = True)

df1['Trans Type'].replace('9.0' , '9' , inplace = True)

df1['Trans Type'].replace('8.0', '8' , inplace = True)

df1['Trans Type'].replace('7.0', '7' , inplace = True)

df1['Trans Type'].replace(['6.0' ,6.0, '6-speed' , 'Tiptronic'] , '6' , inplace = True)

df1['Trans Type'].replace(['5.0',5.0 , '5-speed','5-Speed' , 'HD 5'] , '5' , inplace = True)

df1['Trans Type'].replace(['4-Speed','4.0',4.0] , '4' , inplace = True)

df1['Trans Type'].replace(['4-Speed'] , '4' , inplace = True)

df1['Trans Type'].replace(['1.0','2'],'<3',inplace = True)

df1['Trans Type'].replace('3','<3',inplace = True)

df1['Trans Type'].replace([1,'1'],'<3',inplace = True)
df1['Trans Type'].value_counts()
#sns.swarmplot(x = df1['Trans Type'] ,y= df1['MSRP'])
#DI - Direct Injection , SFI - Sequential fuel injection ,EFI -  Electronic Fuel Injection

df1['Fuel System'].replace(['Gasoline Direct Injection' , 'Direct Injection' , 'Port/Direct Injection' , 

                            'Diesel Direct Injection' ,'NDIS', 'DI' , 'Direct Gasoline Injection' ,'MPFI' , 

                            'Turbocharged DI','FSI','SIDI','SMFI','Turbocharged SMPI','TDI', 

                            'TFSI Direct','Turbocharged','Direct injection','TFSI','GDI','SDI','Turbo-Diesel',

                            'Turbo-Charged OHV','FSI Direct','DOHC FSI Direct','Direct','DISI','SFI/DI'

                            'Turbocharged FSI','Turbo-Charged DI','CDI','Turbocharged FSI','FI','SFI/DI'

                            'NDIS','CRD','DIS','SFI/DI','SFI/DI','PFI','HPI'] , 'DI' , inplace = True)



df1['Fuel System'].replace(['Sequential MPI','SMPI','SEFI','MPI','MFI','SFI FlexFuel','Sequential MPI (injection)'

                            'Sequential Fuel Injection','SPI','SI','Sequential Fuel Injection',

                            'Supercharged SPFI','SMPFI','Sequential MPI (injection)'] , 'SFI' , inplace = True)



df1['Fuel System'].replace(['Electronic Fuel Injection','PGM-FI','Electronic fuel injection' , 

                            'Turbocharged EFI','EMPI','PGM-FI MPI','Supercharged EFI','FFV',

                            'Electronic Fuel Injectino','Electric','Hydrogen','Turbocharged EMFI','EFI',

                            'FFV','SPFI'] , 'Electric FI' , inplace = True)
df1['Fuel System'].value_counts()
sns.countplot(df1['Fuel System'],hue=df1['Category'])
sns.countplot(df1['Fuel System'],hue=df1['Drivetrain'])
plt.figure(figsize=(8,4))

g = sns.barplot(

    data=df1,

    x='Fuel System',

    y='MSRP',

    order = df1.groupby(by = 'Fuel System')['MSRP'].mean().sort_values( ascending=False).index ,

    palette='muted'

)

g.set(ylabel = 'Average Price in dollars')

g.set_title('Car Manufacturers')
pattern1 = re.compile(r'(?i)(auto|hd auto|elec)')                                   #Automatic                   

pattern2 = re.compile(r'(?i)(man|hd man)')                                          #Manual               

pattern3 = re.compile(r'(?i)(cont|cvt|ECVT)')                                       #CVT

pattern4 = re.compile(r'(?i)(Automatic w/OD|Automatic|Auto w/OD|elec|tip|smg)')     #Automatic

pattern5 = re.compile(r'(?i)(Man|dsg|hd)')                                          #Manual



df1['Trans Description Cont.'].fillna('NA',inplace = True)



def transmission(x):

    if re.match(pattern2,x) or re.search(pattern5,x):

        return 'Manual'

    elif re.match(pattern1,x) or re.search(pattern4,x):

        return 'Automatic'

    elif re.match(pattern3,x):

        return 'CVT'

    else:

        return x

    

df1['Trans Description Cont.'] = df1['Trans Description Cont.'].apply(transmission)

df1['Trans Description Cont.'].value_counts()
plt.figure(figsize=(8,4))

g = sns.barplot(

    data=df1,

    x='Trans Description Cont.',

    y='MSRP',

    order = df1.groupby(by = 'Trans Description Cont.')['MSRP'].mean().sort_values( ascending=False).index ,

    palette='muted'

)

g.set(ylabel = 'Average Price in dollars')

g.set_title('Car Manufacturers')
#Net horsepower

df1['SAE Net Horsepower @ RPM'] = pd.to_numeric(df1['SAE Net Horsepower @ RPM'].str.split('@').str[0].str.split(' ').str[0],errors='coerce')
plt.figure(figsize=[20,10])

sns.set(font_scale=1.25)

sns.distplot(df1['SAE Net Horsepower @ RPM'].dropna(),color='g')

plt.xlabel('Horsepower')

plt.ylabel('PDF')
lm = sns.lmplot(x = 'SAE Net Horsepower @ RPM' , 

           y = 'MSRP' ,

           markers= '.',

           col = 'Drivetrain',

           col_wrap=2,

           palette='Blues',

           scatter_kws={'alpha':0.4, 'color':'#fa9943'},

           line_kws={'color': '#db751a'},

           data=df1)

lm.set(ylim=(0, None))

lm.fig.suptitle('Comparision between Horsepower and Price',y=1.05)
#Net Torque

df1['SAE Net Torque @ RPM'] = pd.to_numeric(df1['SAE Net Torque @ RPM'].str.split('@').str[0].str.split(' ').str[0], errors='coerce')
plt.figure(figsize=(20,10))

sns.scatterplot(df1['SAE Net Torque @ RPM'] , df1['MSRP'],hue = df1['Drivetrain'] , alpha = 0.3)
#Torque value above 1000 Nm is highly unlikely

df[df1['SAE Net Torque @ RPM'] > 1000][['Model','Engine','SAE Net Torque @ RPM']]
#Torque produced by Chevrolet Tahoe is around 460 Nm

df1['SAE Net Torque @ RPM'].replace(3350,460,inplace=True)
plt.figure(figsize=(20,10))

sns.scatterplot(df1['SAE Net Torque @ RPM'] , df1['MSRP'],hue = df1['Drivetrain'] , alpha = 0.3)
df1['Displacement'] = pd.to_numeric(df1['Displacement'].str.strip(' ').str.split('L').str[0].str.split('/').str[0].str.split(' ').str[0]

                      ,errors='coerce')
df1['Displacement'].plot(kind='box')
df[df1['Displacement'] > 8][['Engine','Displacement']]
#BMW i3 is a hybrid car which has engine size of 650 cc

df1['Displacement'].replace(39.5,0.65,inplace=True)
g = sns.jointplot(data=df1 , x = 'Displacement' , y = 'MSRP',kind='reg',color = 'g',

                 joint_kws = {'scatter_kws':dict(alpha=0.3)})
df1['Turning Diameter - Curb to Curb (ft)'] = pd.to_numeric(df1['Turning Diameter - Curb to Curb (ft)'].str.split(' ').str[0],

                                                            errors='coerce')
df1['Turning Diameter - Curb to Curb (ft)'].plot(kind='box')
#Outliers

df[(df1['Turning Diameter - Curb to Curb (ft)'] < 20) | (df1['Turning Diameter - Curb to Curb (ft)'] > 80)][['Model' , 'Turning Diameter - Curb to Curb (ft)']]
df1['Turning Diameter - Curb to Curb (ft)'] = df1['Turning Diameter - Curb to Curb (ft)'].apply(lambda x: x*2 if x <20 else x)
plt.figure(figsize=(20,10))

sns.scatterplot(df1['Turning Diameter - Curb to Curb (ft)'] , df1['SAE Net Horsepower @ RPM'],hue = df1['Category'] , alpha = 0.3)
df1['Front Wheel Material'].replace('Styled Steel','Steel',inplace = True)

df1['Front Wheel Material'].replace('Forged Aluminum','Aluminum',inplace = True)

df1['Front Wheel Material'].replace('Chrome','Alloy',inplace= True)

df1['Front Wheel Material'].value_counts()
#df1['Front tire vehicle type']=list(map(lambda x: str(x)[0],df1['Front Tire Size']))

df1['Front tire width']=list(map(lambda x: str(x)[1:4],df1['Front Tire Size']))

df1['Front tire aspect ratio']=list(map(lambda x: str(x)[0:2],df1['Front Tire Size'].str.split('/').str[1]))

df1['Front tire speed ratings/cons.type']=list(map(lambda x: str(x)[2:-2],df1['Front Tire Size'].str.split('/').str[1]))

df1['Front tire rim size']=list(map(lambda x: str(x)[-2:],df1['Front Tire Size'].str.split('/').str[1]))



df1=df1.drop('Front Tire Size',axis=1)
df1['Front tire aspect ratio'] = df1['Front tire aspect ratio'].replace(['YR','na',''],np.nan)

df1['Front tire aspect ratio'] = df1['Front tire aspect ratio'].replace('71',70)

df1['Front tire aspect ratio'] = df1['Front tire aspect ratio'].replace('31',30)
#df1['Front tire aspect ratio'].value_counts(dropna = False)
pattern1 = re.compile(r'(?i)(z)')     #Z - 240+ Kmph                  

pattern2 = re.compile(r'(?i)(v)')     #V - 240  Kmph       

pattern3 = re.compile(r'(?i)(h)')     #H - 210  Kmph

pattern4 = re.compile(r'(?i)(t)')     #T - 190  Kmph

pattern5 = re.compile(r'(?i)(r)')     #R - 170  Kmph



def tire_speed(x):

    if re.match(pattern1,x):

        return 'Z'

    elif re.match(pattern2,x):

        return 'V'

    elif re.match(pattern3,x):

        return 'H'

    elif re.match(pattern4,x):

        return 'T'

    elif re.match(pattern5,x):

        return 'R'

    else:

        return 'NA'

    

df1['Front tire speed ratings/cons.type'] = df1['Front tire speed ratings/cons.type'].apply(tire_speed)

df1['Front tire speed ratings/cons.type'].value_counts(dropna = False)
df1['Front tire rim size'].value_counts(dropna = False)
pattern1 = re.compile(r'(1[5-9]|2[0-8])')



def rim_size(x):

    if re.match(pattern1,x):

        return x

    else:

        return np.nan

    

df1['Front tire rim size'] = df1['Front tire rim size'].apply(rim_size)

df1['Front tire rim size'].value_counts(dropna = False)
df1['Stabilizer Bar Diameter - Front (in)'] = pd.to_numeric(df1['Stabilizer Bar Diameter - Front (in)'],errors='coerce')
df1['Stabilizer Bar Diameter - Front (in)'].plot(kind= 'box')
#Above 2 inch is highly unlikely

df1.loc[df1['Stabilizer Bar Diameter - Front (in)']>2,'Stabilizer Bar Diameter - Front (in)'] = np.nan
df1['Stabilizer Bar Diameter - Front (in)'].plot(kind= 'box')
y_n = ['Air Bag-Frontal-Driver', 'Air Bag-Frontal-Passenger',

       'Air Bag-Passenger Switch (On/Off)', 'Air Bag-Side Body-Front',

       'Air Bag-Side Body-Rear', 'Air Bag-Side Head-Front',

       'Air Bag-Side Head-Rear', 'Brakes-ABS', 'Child Safety Rear Door Locks',

       'Daytime Running Lights', 'Traction Control', 'Night Vision',

       'Rollover Protection Bars', 'Fog Lamps', 'Parking Aid',

       'Tire Pressure Monitor', 'Back-Up Camera', 'Stability Control']



for i in y_n:

    df1[i] = df1[i].map({'Yes':1 , 'No':0})
df1['Corrosion Miles/km'] = df1['Corrosion Miles/km'].str.replace(",", "").str.replace("Unlimited", "150000")

df1['Corrosion Miles/km'] = df1['Corrosion Miles/km'].astype(float)

df1['Corrosion Miles/km'].value_counts()
df1['Drivetrain Miles/km'] = df1['Drivetrain Miles/km'].str.replace(",", "").str.replace("Unlimited", "150000")

df1['Drivetrain Miles/km'] = df1['Drivetrain Miles/km'].astype(float)

df1['Drivetrain Miles/km'].value_counts()
df1['Basic Miles/km'] = df1['Basic Miles/km'].str.replace(",", "").str.replace("Unlimited", "150000")

df1['Basic Miles/km'] = df1['Basic Miles/km'].str.replace("49999", "50000")

df1['Basic Miles/km'] = df1['Basic Miles/km'].astype(float)

df1['Basic Miles/km'].value_counts()
df1['Roadside Assistance Miles/km'] = df1['Roadside Assistance Miles/km'].str.replace(",", "").str.replace("Unlimited", "150000")

df1['Roadside Assistance Miles/km'] = df1['Roadside Assistance Miles/km'].str.replace("49711", "50000")

df1['Roadside Assistance Miles/km'] = df1['Roadside Assistance Miles/km'].str.replace("24000", "25000")

df1['Roadside Assistance Miles/km'] = df1['Roadside Assistance Miles/km'].astype(float)

df1['Roadside Assistance Miles/km'].value_counts()
df1['Drivetrain Years'] = df1['Drivetrain Years'].str.replace('Unlimited','20')

df1['Roadside Assistance Years'] = df1['Roadside Assistance Years'].str.replace('Unlimited','20')
df1.drop(columns=['EPA Classification' , 'Style Name' ,'Body Style' ,'Transmission','Steering Type','Brake Type'],inplace = True)
df1.head()
#df1.to_csv('New_cars_cleaned.csv',index=False)
missing_values(df1)
cat_col_imp = ['Drivetrain' , 'Category' , 'Engine' , 'Trans Type']
df1.dropna(axis=0 , how = 'any' , subset=['MSRP','Drivetrain'] , inplace=True)

df1[['MSRP' , 'Drivetrain']].isnull().sum()
#Imuting missing values in egnine by grouping them by manufacturer and category wise

df1['Engine'] = df1['Engine'].fillna(df1.groupby(['Manufacturer', 'Category'])['Engine'].transform(lambda x: x.value_counts().idxmax()))
#Trans type

df1['Trans Type'].value_counts(dropna=False)
##Imuting missing values in transmission type by grouping them by manufacturer and engine wise

df1['Trans Type'] = df1['Trans Type'].fillna(df1.groupby(['Manufacturer','Category','Engine'])['Trans Type'].transform(lambda x: x.value_counts().idxmax()))
df1['Trans Description Cont.'].value_counts(dropna= False)
df1['Trans Description Cont.'] = df1['Trans Description Cont.'].fillna(df1.groupby(['Manufacturer','Category','Engine'])['Trans Description Cont.'].transform(lambda x: x.value_counts().idxmax()))
df1['Fuel System'].value_counts(dropna=False)
df['Fuel System'] = df['Fuel System'].fillna('Unknown')
#Remaining columns tht has to be imputed

num_col_imp = ['EPA Fuel Economy Est - City (MPG)', 'Base Curb Weight (lbs)',

               'Passenger Volume (ft³)', 'Wheelbase (in)', 'Track Width, Front (in)',

               'Height, Overall (in)', 'Fuel Tank Capacity, Approx (gal)',

               'SAE Net Torque @ RPM', 'Fuel System', 'SAE Net Horsepower @ RPM',

               'Displacement', 'Basic Miles/km',

               'Basic Years', 'Corrosion Miles/km', 'Corrosion Years',

               'Drivetrain Miles/km', 'Drivetrain Years',

               'Turning Diameter - Curb to Curb (ft)']
#One hot encoding

#If a category has more than 300 values then a new column is created



cat_col = ['Engine' , 'Drivetrain' , 'Trans Description Cont.' , 'Fuel System','Suspension Type - Front',

           'Trans Type','Suspension Type - Rear','Manufacturer','Category','Front Wheel Material','Front tire speed ratings/cons.type']

print('The Encoding is applied for: ')

for col in cat_col:

    freqs=df1[col].value_counts()

    k=freqs.index[freqs>300]

    for cat in k:

        name=col+'_'+cat

        df1[name]=(df1[col]==cat).astype(int)

    del df1[col]

    print(col)
df1.head()
for i in list(df1.columns[df1.dtypes == 'O']):

    df1[i] = pd.to_numeric(df1[i],errors='coerce')
# df2 = df1.copy()

# df2 = df2[:1000]   

# #MICE IMPUTATION

# from fancyimpute import IterativeImputer

# MICE_imputer = IterativeImputer(verbose=2)



# df2.iloc[:,:] = MICE_imputer.fit_transform(df1)



# df2.to_csv('Car_MICE_imp.csv',index=False)
df3 = df1.copy()

df3 = df3[:5000]


from fancyimpute import KNN



# Initialize KNN

knn_imputer = KNN(verbose = 2)



# Impute using fit_tranform on diabetes_knn_imputed

df3.iloc[:, :] = knn_imputer.fit_transform(df3)
# df2.to_csv('Car_MICE_imp.csv',index=False)
# mice = pd.read_csv('Car_MICE_imp.csv')

# mice.head()
#knn = pd.read_csv('Car_KNN_imp.csv')

df3.head()