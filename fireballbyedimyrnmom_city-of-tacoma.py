# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

CODE_ENFORCEMENT_VIOLATIONS = pd.read_csv("../input/CODE_ENFORCEMENT_VIOLATIONS.csv")

TACOMA_BUSINESS_LICENSE = pd.read_csv("../input/TACOMA_BUSINESS_LICENSE.csv")

Tacoma_Crime = pd.read_csv("../input/Tacoma_Crime.csv")

Tacoma_Traffic_Collision_Data_2010_ = pd.read_csv("../input/Tacoma_Traffic_Collision_Data_2010-.csv")
CODE_ENFORCEMENT_VIOLATIONS.head(2) 
CODE_ENFORCEMENT_VIOLATIONS.shape
list(CODE_ENFORCEMENT_VIOLATIONS)
#rename and update the df

codeV= CODE_ENFORCEMENT_VIOLATIONS.drop(['CASE NUMBER','COUNT', 'ASSESSOR INFORMATION', 'NUMBER OF DAYS FROM OPEN TO CLOSED', 'CURRENT STATUS','PROPERTY OWNER','HISTORIC DISTRICT', 'COMMUNITY GROUP', 'DATE UPDATED', 'INSPECTOR'], axis=1)

codeV.head(2)
codeV.info()
#choosing specific columns to work with

cViolations=codeV[['OPEN DATE',  'ADDRESS', 'New Georeferenced Column', 'DESCRIPTION', 'CASE TYPE', 'SECTOR AND DISTRICT']]

cViolations.head(3)
TACOMA_BUSINESS_LICENSE.head(2) 
TACOMA_BUSINESS_LICENSE.shape
Bizns=TACOMA_BUSINESS_LICENSE.drop(['LICENSE NUMBER','OWNER NAME', 'P.O. BOX', 'NAICS CODE', 'BUSINESS OPEN DATE'], axis=1)

Bizns.head(2)
Tacoma_Crime.head(3)
Tacoma_Crime.shape
Tacoma_Crime.info()
Tacoma_Traffic_Collision_Data_2010_.head(2)
Tacoma_Traffic_Collision_Data_2010_.shape
list(Tacoma_Traffic_Collision_Data_2010_)
#rename and pick only columns desired



TrafikBoom= Tacoma_Traffic_Collision_Data_2010_[['DATE', '24 Hr Time', 'MOST SEVERE INJURY TYPE',  'TOTAL FATALITIES', 

                                                 'Collision Involving Pedestrian or Bicyclist ', 'WEATHER',  

                                                 'ROAD SURFACE CONDITIONS', 'Hit and Run', 'Most Severe sobriety type', 

                                                 'Year', 'New Georeferenced Column', 'Junction Relationship']]

TrafikBoom.head(2)
Tacoma_Crime.head(2)
##convert specific data to numerical for statistical comparisson.

Crime=Tacoma_Crime[['Crime', 'Approximate Time']]

Crime.head(2)
import numpy as np 

##Find NaNs

# Total number of missing values

Crime.isnull().sum().sum()

###There is no missing data
#List possible labels for crime.

Crime['Crime'].unique() 

#There are over 30 labels.
#LABEL-ENCODE



from sklearn.preprocessing import LabelEncoder



#Converts the object column into category type

Crime['Approximate Time']=Crime['Approximate Time'].astype('category')

Crime['Approximate Time']=Crime['Approximate Time'].cat.codes

Crime.head(3)
#when group by categories, the rows reduce but still dificult to observe as is.

gByTime = Crime.groupby('Approximate Time') 

gByTime.head()
#Converts the object column into category type

Crime['Crime']=Crime['Crime'].astype('category')

Crime['CrimeCats']=Crime['Crime'].cat.codes

Crime.head(3)
#Statistical description 

Crime.describe()
Crime.median()
Crime['Approximate Time'].astype(float)
Crime['CrimeCats'].astype(float)
#correlation

Crime.corr()
n = {'All Other Larceny' :0, 'Theft From Motor Vehicle': 1,

       'False Pretenses/Swindle/Confidence Game': 2,

       'Stolen Property Offenses': 0,

       'Destruction/Damage/Vandalism of Property': 3, 'Shoplifting': 0,

       'Burglary/Breaking & Entering': 4, 'Simple Assault': 5,

       'Theft of Motor Vehicle Parts/Accessories': 1, 'Motor Vehicle Theft': 1,

       'Intimidation':6, 'Pocket-Picking': 0, 'Impersonation': 2, 'Arson': 7,

       'Robbery': 0, 'Aggravated Assault': 5,

       'Credit Card/Automatic Teller Fraud': 2, 'Weapon Law Violations': 8,

       'Drug/narcotic Violations': 8,

       'Violation of No Contact/Protection Order': 6,

       'Pornography/Obscene Material': 10, 'Embezzlement': 2, 'Wire Fraud': 2,

       'Counterfeiting/Forgery': 2, 'Extortion/Blackmail':6, 'Purse-Snatching': 0,

       'Theft From Building': 0, 'Drug Equipment Violations': 8,

       'Kidnaping/Abduction': 5, 'Prostitution': 10,

       'Theft From Coin Operated Machine or Device': 0,

       'Murder and Nonnegligent Manslaughter': 9,

       'Assisting or Promoting Prostitution': 10, 'Welfare Fraud': 2,

       'Negligent Manslaughter': 9, 'Identity Theft': 2,

       'Human Trafficking/Commercial Sex Acts':10, 'Justifiable Homicide': 9}

Crime['Crime'] =Crime['Crime'].replace(n)

Crime.head(3)
corr1=Crime.drop(['CrimeCats'], axis=1)

corr1.head(2)
corr1.describe()
#The 10 catgeories (instead of 38) did not make much difference in correlation between hour of the day and crime type.

corr1.corr()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Density Plot and Histogram of times of crimes

sns.distplot(Crime['Approximate Time'], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
##Logistic regression

#Working with a different df

TrafikBoom.head(3)    
#I verified for NaN in the column Hit and Run

TrafikBoom['Hit and Run'].isnull().sum().sum()
#Trying to find the NaN

TrafikBoom[TrafikBoom['Hit and Run'].isnull()]

##This entire row is a NaN.
TrafikBoom.drop(TrafikBoom.index[154], inplace=True)

#I verified for NaN in the column Hit and Run

TrafikBoom['Hit and Run'].isnull().sum().sum()
#Create dummy variables for True or False

X_enc = pd.get_dummies(TrafikBoom, columns=['Hit and Run']) 
#a binary df 

BiHnR= X_enc[['TOTAL FATALITIES', 'Year', 'Hit and Run_False', 'Hit and Run_True']]

BiHnR.head(2)
BiHnR.info()
#Regression Plot

import seaborn as sns

sns.regplot(x='Hit and Run_False', y='Hit and Run_True', data=BiHnR, logistic=True)

BiHnR.astype(int)
#clear correlation between true and false

BiHnR.corr()
BiHnR.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(BiHnR.drop('Hit and Run_True',axis=1), 

           BiHnR['Hit and Run_True'], test_size=0.30, random_state=0)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

model
model.fit(X_train, y_train)
#predictions

Predictions = model.predict(X_test)

Predictions
from sklearn.metrics import classification_report

print(classification_report(y_test,Predictions))
model.intercept_
model.coef_
model.score(X_train, y_train)
model.score(X_test, y_test)
Bizns.head(2)
#Column as df

S=Bizns[['NAICS CODE DESCRIPTION']]
#As an array

B=S['NAICS CODE DESCRIPTION']
S.astype(str)

S=S.replace("Limited-Service Restaurants", "Restaurant")

S=S.replace("Full-Service Restaurants", "Restaurant")
List1=B.unique()

List1

#There are over 800 labels
S.astype(str)

S=S.replace("Beer Wine and Liquor Stores", "Liquor Store")

S=S.replace("Drinking Places (Alcoholic Beverages)  (ES 0200 0150  08)", "Drinking Places")

S=S.replace("Other Gambling Industries", "Glambing")
##group some unused labels into the name Other

S.astype(str)

S=S.replace("Taxi Service", "Other")

S=S.replace("Landscaping Services", "Other")

S=S.replace("Lessors of Residential Buildings and Dwellings", "Other")

S=S.replace("Offcs of Mental Hlth Practitioners (ex Physicians) (ES 0200", "Other")

S=S.replace("Independent Artists Writers and Performers (ES 0200 0225", "Other")

S=S.replace("Electrical Contractors   (ES 0200 0225    06)", "Other")

S=S.replace("All Other Professional/Scientific/Technical Services (ES 020", "Other")
S=S.replace("Lessors of Nonresidential Bildngs (ex Miniwrhss) (ES 0200 02", "Other")

S=S.replace("Residential Remodelers  (ES 0200 0225  08)", "Other")

S=S.replace("New Single-Family Housing Constn (ex Operative Bldrs) (ES 02", "Other")

S=S.replace("Plumbing/Heating/Air-Conditioning Contractors (ES 0200 0225", "Other")

S=S.replace("Broadwoven Fabric Mills", "Other")

S=S.replace("Industrial Mold Manufacturing  (ES 0200 0225    08)", "Other")
#Identify desired labels

S=S.replace("Hotels (except Casino Hotels) and Motels  (ES 0500 0450  08)", "Hotel")

S=S.replace("Child Day Care Services", "Day Care")

S=S.replace("Colleges Universities and Professional Schools", "Colleges")

S=S.replace("Elementary and Secondary Schools", "Grade schools")
S=S.replace("Beauty Salons  (ES 0200 0225    08)", "Other")

S=S.replace("Janitorial Services  (ES 0200 0225  06)", "Other")

S=S.replace("All Othr Miscll Store Retailers (ex Tobacco Strs) (ES 0200 0", "Other")

S=S.replace("Dry Pasta Dough and Flour Mixes Manufacturing from Purchas", "Other")

S=S.replace("Offices of Real Estate Agents and Brokers   (ES 0200 0225", "Other")

S=S.replace("All Other Specialty Trade Contractors (ES 0200 0225  08)", "Other")

S=S.replace("Offices of Lawyers", "Other")

S=S.replace("Cement Manufacturing", "Other")

S=S.replace("Fiber Yarn and Thread Mills", "Other")
S=S.replace("Commercial and Institutional Building Construction  (ES  020", "Other")

S=S.replace("Painting and Wall Covering Contractors   (ES 0200 0225    06", "Other")

S=S.replace("Wood Preservation  (ES 0200 0225  08)", "Other")

S=S.replace("Security Guards and Patrol Services", "Other")

S=S.replace("Other Personal Care Services   (ES 0200 0225    08)", "Other")

S=S.replace("Offices of All Other Miscellaneous Health Practitioners", "Other")

S=S.replace("Administrative Management and General Management Consulting", "Other")

S=S.replace("Electroplating/Plating/Polishing/Anodizing/Coloring(ES 0200", "Other")

S=S.replace("Other Personal Care Services   (ES 0200 0225    08)", "Other")

S=S.replace("Direct Mail Advertising", "Other")

S=S.replace("News Dealers and Newsstands  (ES 0200 0225    08)", "Other")
S=S.replace("Site Preparation Contractors (ES 0200 0225    08)", "Other")

S=S.replace("All Other Personal Services   (ES 0200 0225    08)", "Other")

S=S.replace("Plastics Bottle Manufacturing", "Other")

S=S.replace("Conveyor and Conveying Equipment Manufacturing", "Other")

S=S.replace("Fluid Milk Manufacturing  (0200 0225  08)", "Other")

S=S.replace("Offices of Physicians (ex Mental Health Specialists) (ES 020", "Other")

S=S.replace("Roofing Contractors   (ES 0200 0225    06)", "Other")

S=S.replace("Security Systems Services (except Locksmiths)  (ES 0200 0225", "Other")

S=S.replace("Music Publishers", "Other")

S=S.replace("Deep Sea Freight Transportation", "Other")

S=S.replace("Relay and Industrial Control Manufacturing", "Other")
S=S.replace("Poured Concrete Foundation/Structure Contractor  (ES 0200 02", "Other")

S=S.replace("Finish Carpentry Contractors (ES 0200 0225    08)", "Other")

S=S.replace("General Freight Trucking/Long-Distance/Truckload (ES 0200 02", "Other")

S=S.replace("Hardwood Veneer and Plywood Manufacturing", "Other")

S=S.replace("Masonry Contractors", "Other")

S=S.replace("Abrasive Product Manufacturing", "Other")

S=S.replace("Electronic Coil Transformer and Other Inductor Manufacturi", "Other")

S=S.replace("Engineering Services   (ES 0200 0225    08)", "Other")

S=S.replace("Other Direct Selling Establishments  (ES 0200 0225    08)", "Other")

S=S.replace("Electronics Stores", "Other")

S=S.replace("Other Nonferrous Metal Foundries (except Die-Casting)", "Other")

S=S.replace("Overhead Traveling Crane Hoist and Monorail System Manufac", "Other")
S=S.replace("Other Individual and Family Services   (ES 0200 0225    08)", "Other")

S=S.replace("Photography Studios Portrait   (ES 0200 0150   08)", "Other")

S=S.replace("Flooring Contractors   (ES 0200 0225    06)", "Other")

S=S.replace("Media Representatives", "Other")

S=S.replace("Fertilizer (Mixing Only) Manufacturing", "Other")

S=S.replace("International Affairs", "Other")

S=S.replace("Interurban and Rural Bus Transportation", "Other")

S=S.replace("Regulation and Administration of Communications Electric G", "Other")
S=S.replace("General Automotive Repair   (ES 0200 0225    08)", "Other")

S=S.replace("Custom Computer Programming Services (ES 0200 0225   08)", "Other")

S=S.replace("Wholesale Trade Agents and Brokers  (ES 0200 0225    08)", "Other")

S=S.replace("Other Depository Credit Intermediation", "Other")

S=S.replace("Soybean Farming", "Other")

S=S.replace("Regulation and Administration of Transportation Programs", "Other")

S=S.replace("Credit Card Issuing   (ES 0200 0225    08)", "Other")

S=S.replace("Lime Manufacturing", "Other")

S=S.replace("Snack and Nonalcoholic Beverage Bars", "Other")

S=S.replace("General Freight Trucking Local  (ES 0200 0225    08)", "Other")

S=S.replace("Other Basic Inorganic Chemical Manufacturing", "Other")

S=S.replace("Insurance Agencies and Brokerages   (ES 0200 0225    08)", "Other")

S=S.replace("Computer Terminal and Other Computer Peripheral Equipment Ma", "Other")
S=S.replace("Packing and Crating", "Other")

S=S.replace("Offices of Dentists", "Other")

S=S.replace("Insurance Agencies and Brokerages   (ES 0200 0225    08)", "Other")

S=S.replace("Beet Sugar Manufacturing", "Other")

S=S.replace("Custom Roll Forming", "Other")

S=S.replace("Household Furniture (except Wood and Metal) Manufacturing", "Other")

S=S.replace("Other Apparel Knitting Mills", "Other")

S=S.replace("Architectural Services", "Other")

S=S.replace("Graphic Design Services", "Other")

S=S.replace("Tire Retreading  (ES 0200 0225  08)", "Other")

S=S.replace("Fluid Power Pump and Motor Manufacturing", "Other")

S=S.replace("Asphalt Shingle and Coating Materials Manufacturing (ES 0200", "Other")
S=S.replace("Wired Telecommunications Carriers   (ES 0200 0225    08)", "Other")

S=S.replace("Commercial & Industrial Machinery & Equipment (expt Automoti", "Other")

S=S.replace("Tortilla Manufacturing", "Other")

S=S.replace("Ophthalmic Goods Manufacturing  (ES 0200 0225    08)", "Other")

S=S.replace("Directory and Mailing List Publishers (ES 0200 0150  08)", "Other")

S=S.replace("Farm Product Warehousing and Storage", "Other")

S=S.replace("Other Building Material Dealers  (ES 0200 0225    08)", "Other")

S=S.replace("Other Commercial and Industrial Machinery and Equipment Rent", "Other")

S=S.replace("Gasket Packing and Sealing Device Manufacturing", "Other")

S=S.replace("Specialty Canning", "Other")

S=S.replace("Optical Instrument and Lens Manufacturing", "Other")

S=S.replace("Cheese Manufacturing", "Other")

S=S.replace("Meat Processed from Carcasses (ES 0200 0225  08)", "Other")
S=S.replace("Other Activities Related to Real Estate", "Other")

S=S.replace("Medical Dental and Hospital Equipment and Supplies Merchan", "Other")

S=S.replace("Electric Power Distribution", "Other")

S=S.replace("Paper (except Newsprint) Mills (ES 0200 0225  08)", "Other")

S=S.replace("Nonchocolate Confectionery Manufacturing (ES 0800 0400 08)", "Other")

S=S.replace("Residential Property Managers", "Other")

S=S.replace("Marketing Consulting Services  (ES 0200 0225    08)", "Other")

S=S.replace("Laminated Plastics Plate Sheet (except Packaging) and Shap", "Other")

S=S.replace("Farm Management Services", "Other")

S=S.replace("Rice Milling", "Other")

S=S.replace("Formal Wear and Costume Rental  (ES 0200 0225    08)", "Other")

S=S.replace("Dental Equipment and Supplies Manufacturing", "Other")
S=S.replace("All Other Business Support Services", "Other")

S=S.replace("Other Management Consulting Services", "Other")

S=S.replace("Irradiation Apparatus Manufacturing", "Other")

S=S.replace("Primary Battery Manufacturing", "Other")

S=S.replace("Goat Farming", "Other")

S=S.replace("Other Communication and Energy Wire Manufacturing", "Other")
#Other desired labels

S=S.replace("Short-Line Railroads", "Railroads")
S=S.replace("Industrial Machinery/Equipment Merch Whlslrs (ES 0200 0225", "Other")

S=S.replace("Other Services to Buildings and Dwellings", "Other")

S=S.replace("Construction Machinery Manufacturing", "Other")

S=S.replace("Automobile Manufacturing", "Other")

S=S.replace("Metal Can Manufacturing", "Other")

S=S.replace("Charter Bus Industry", "Other")

S=S.replace("Used Merchandise Stores (ES 0200 0225    08)", "Other")

S=S.replace("Other Scientific and Technical Consulting Services  (ES 0200", "Other")

S=S.replace("Plumbing Fixture Fitting and Trim Manufacturing", "Other")

S=S.replace("Hay Farming", "Other")

S=S.replace("Printing Ink Manufacturing", "Other")

S=S.replace("Petrochemical Manufacturing", "Other")

S=S.replace("Upholstered Household Furniture Manufacturing  (ES 0200 0225", "Other")
S=S.replace("Drywall and Insulation Contractors   (ES 0200 0225    06)", "Other")

S=S.replace("Other Misc Nondurable Goods Merch Whlslrs (ES 0200 0225  08)", "Other")

S=S.replace("Inland Water Passenger Transportation", "Other")

S=S.replace("Commodity Contracts Brokerage", "Other")

S=S.replace("Fiber Optic Cable Manufacturing", "Other")

S=S.replace("Motorcycle Bicycle and Parts Manufacturing", "Other")

S=S.replace("Berry (except Strawberry) Farming", "Other")
S=S.replace("Other Accounting Services  (ES 0200 0225    08)", "Other")

S=S.replace("Othr Personal/Hshld Goods Repair/Maintenance (ES 0200 0225", "Other")

S=S.replace("Manufactured (Mobile) Home Dealers", "Other")

S=S.replace("Metal Crown Closure and Other Metal Stamping (except Autom", "Other")

S=S.replace("Adhesive Manufacturing", "Other")

S=S.replace("Parking Lots and Garages   (ES 0200 0225    08)", "Other")

S=S.replace("Barber Shops  (ES 0200 0225    08)", "Other")

S=S.replace("Iron and Steel Pipe and Tube Manufacturing from Purchased St", "Other")

S=S.replace("Forest Nurseries and Gathering of Forest Products", "Other")

S=S.replace("Other Snack Food Manufacturing", "Other")
S=S.replace("Family Clothing Stores", "Other")

S=S.replace("Other Noncitrus Fruit Farming", "Other")

S=S.replace("Small Electrical Appliance Manufacturing", "Other")

S=S.replace("Other Vegetable (except Potato) and Melon Farming", "Other")

S=S.replace("Mattress Manufacturing", "Other")

S=S.replace("Flavoring Syrup and Concentrate Manufacturing", "Other")

S=S.replace("Civic and Social Organizations  (ES 0200 0225  06)", "Other")
S=S.replace("Mobile Food Services", "Restaurant")
S=S.replace("Electronic Shopping  (ES 0200 0225    08)", "Other")

S=S.replace("International Trade Financing", "Other")

S=S.replace("Residential Electric Lighting Fixture Manufacturing", "Other")

S=S.replace("Precision Turned Product Manufacturing", "Other")

S=S.replace("Photofinishing Laboratories (except One-Hour)  (ES 0200 0150", "Other")

S=S.replace("Gift Novelty and Souvenir Stores", "Other")
S=S.replace("Supermarkets/Othr Grocery (ex Conven) Stores (ES 0400 0300", "Other")

S=S.replace("All Other Miscellaneous Schools/ Instruction (ES 0200 0225", "Other")

S=S.replace("Polystyrene Foam Product Manufacturing", "Other")

S=S.replace("Aluminum Sheet Plate and Foil Manufacturing", "Other")

S=S.replace("Computer Systems Design Services  (ES 0200 0225    08)", "Other")
S=S.replace("Used Car Dealers", "Other")

S=S.replace("Fitness and Recreational Sports Centers", "Other")

S=S.replace("Securities and Commodity Exchanges", "Other")

S=S.replace("Noncurrent-Carrying Wiring Device Manufacturing", "Other")

S=S.replace("Motor Vehicle Body Manufacturing", "Other")

S=S.replace("Hydroelectric Power Generation   (ES 0200 0225    08)", "Other")
S=S.replace("Other Computer Related Services", "Other")

S=S.replace("Temporary Help Services   (ES 0200 0225    08)", "Other")

S=S.replace("Support Activities for Coal Mining", "Other")

S=S.replace("Motor Vehicle Gasoline Engine and Engine Parts Manufacturing", "Other")

S=S.replace("Caterers", "Other")

S=S.replace("Highway Street and Bridge Construction  (ES 0200 0225    0", "Other")
S=S.replace("Offices of Physical Occupational and Speech Therapists and", "Other")

S=S.replace("Educational Support Services   (ES 0200 0225    08)", "Other")

S=S.replace("Zoos and Botanical Gardens", "Other")

S=S.replace("Paperboard Mills  (ES 0800 0400  08)", "Other")

S=S.replace("All Other Leather Good and Allied Product Manufacturing", "Other")

S=S.replace("Breakfast Cereal Manufacturing", "Other")
S=S.replace("Pet Care (except Veterinary) Services  (ES 0200 0225  06)", "Other")

S=S.replace("Investment Advice   (ES 0200 0225    08)", "Other")

S=S.replace("Other Insurance Funds", "Other")

S=S.replace("Fats and Oils Refining and Blending", "Other")

S=S.replace("Other Support Activities for Air Transportation (ES 0200 022 ", "Other")
S=S.replace("All Other Health and Personal Care Stores  (ES 0200 0225", "Other")

S=S.replace("Environmental Consulting Services", "Other")

S=S.replace("Other Fabricated Wire Product Manufacturing  (ES 0200 0225", "Other")

S=S.replace("Glass Container Manufacturing", "Other")

S=S.replace("Electrical Contractors", "Other")

S=S.replace("Iron and Steel Forging   (ES 0200 0225    08)", "Other")
S=S.replace("Other Grocery/Related Products Merch Whlslrs (ES 0200 0225", "Other")

S=S.replace("Translation and Interpretation Services", "Other")

S=S.replace("Secondary Market Financing", "Other")

S=S.replace("Other Electronic Component Manufacturing", "Other")

S=S.replace("Other Aircraft Parts/Auxiliary Equip Manuf (ES 0200 0225  08", "Other")
S=S.replace("Promoters of Performing Arts Sports and Similar Events wit", "Other")

S=S.replace("Art Dealers  (ES 0200 0225    08)", "Other")

S=S.replace("Other Paperboard Container Manufacturing", "Other")

S=S.replace("Crushed and Broken Limestone Mining and Quarrying", "Other")

S=S.replace("Lessors of Nonresidential Buildings (except Miniwarehouses)", "Other")

S=S.replace("Other Metal Valve and Pipe Fitting Manufacturing  (ES 0200 0", "Other")

S=S.replace("Telecommunications Resellers", "Other")

S=S.replace("Nail Salons   (ES 0200 0225    08)", "Other")
S=S.replace("Psychiatric and Substance Abuse Hospitals   (ES 0200 0225", "Hospital")

S=S.replace("Gasoline Stations with Convenience Stores  (ES 0200 0225", "Convenience store") 
S=S.replace("Timber Tract Operations", "Other")

S=S.replace("Books Printing", "Other")

S=S.replace("Sanitary Paper Product Manufacturing (ES 0200 0225  08)", "Other")

S=S.replace("Farm Labor Contractors and Crew Leaders", "Other")

S=S.replace("Mushroom Production", "Other")

S=S.replace("Tire Manufacturing (except Retreading)", "Other")

S=S.replace("Men's and Boys' Cut and Sew Apparel Manufacturing", "Other")

S=S.replace("Publishing Industries (except Internet)", "Other")

S=S.replace("Parole Offices and Probation Offices", "Other")
S=S.replace("Siding Contractors  (ES 0200 0225  08)", "Other")

S=S.replace("Cosmetics Beauty Supplies and Perfume Stores", "Other")

S=S.replace("Offices of Chiropractors", "Other")

S=S.replace("Medicinal and Botanical Manufacturing", "Other")

S=S.replace("Tile and Terrazzo Contractors (ES 0200 0225    08)", "Other")
S=S.replace("Sporting Goods Stores", "Other")

S=S.replace("Water/Sewer Line/ Related Structures Construction (ES 0200", "Other")

S=S.replace("Other Building Finishing Contractors", "Other")

S=S.replace("Glass and Glazing Contractors  (ES 0200 0225  08)", "Other")

S=S.replace("Drugs and Druggists' Sundries Merchant Wholesalers", "Other")

S=S.replace("Interior Design Services  (ES 0200 0225    08)", "Other")

S=S.replace("Local Messengers and Local Delivery", "Other")

S=S.replace("Electrical Apparatust Wiring Supplies & Related Equipt Mer", "Other")

S=S.replace("Automotive Parts and Accessories Stores  (ES 0200 0225    08", "Other")
S=S.replace("Convenience Stores", "Convenience store")

S=S.replace("All Othr Amusement/Recreation Industries (ES 0200 0225  08)", "Amusement")
S=S.replace("Wireless Telecommunications Carriers (except Satellite)", "Other")

S=S.replace("Office Administrative Services", "Other")

S=S.replace("Home Health Care Services   (ES 0200 0225    08)", "Other")

S=S.replace("Specialized Freight (ex Used Goods) Trucking/Local (ES 0200", "Other")

S=S.replace("Jewelry Stores", "Other")

S=S.replace("Sign Manufacturing  (ES 0200 0225    08)", "Other")

S=S.replace("Exterminating and Pest Control Services   (ES 0200 0225    0", "Other")

S=S.replace("Furniture Stores  (ES 0200 0225    08)", "Other")
S=S.replace("Other Misc Durable Goods Merch Wholesalers (ES 0200 0225  08", "Other")

S=S.replace("Auto Body/Paint/Interior Repair/Maintenance (ES 0200 0225  0", "Other")

S=S.replace("Motion Picture and Video Production", "Other")

S=S.replace("Masonry Contractors (ES 0200 0225  08)", "Other")

S=S.replace("All Other Miscellaneous Manufacturing  (ES 0200 0225    08)", "Other")

S=S.replace("Lumber/Plywd/Millwork/Wd Panel Merch Whlslrs (ES 0200 0225", "Other")

S=S.replace("Offices of Certified Public Accountants   (ES 0200 0225    0", "Other")

S=S.replace("All Other Support Services", "Other")

S=S.replace("All Other Transit and Ground Passenger Transportation", "Other")
S=S.replace("Framing Contractors  (ES 0200 0225  08)", "Other")

S=S.replace("Other Building Equipment Contractors  (ES 0200 0225    08)", "Other")

S=S.replace("Tax Preparation Services   (ES 0200 0225    08)", "Other")

S=S.replace("Floor Covering Stores", "Other")

S=S.replace("Landscape Architectural Services  (ES 0200 0225  04)", "Other")

S=S.replace("Commercial Photography  (ES 0200 0225    08)", "Other")

S=S.replace("Other Support Activities for Road Transportation", "Other")

S=S.replace("Other Clothing Stores", "Other")
S=S.replace("Homes for the Elderly  (ES 0200 0225  06)", "Other")

S=S.replace("Other Chemical/Allied Products Merch Whlslrs (ES 0200 0225", "Other")

S=S.replace("All Other Legal Services", "Other")

S=S.replace("Carpet and Upholstery Cleaning Services", "Other")

S=S.replace("Outpatient Mental Hlth/Substance Abuse Centers (ES 0200 0225", "Other")

S=S.replace("Other Heavy and Civil Engineering Construction  (ES 0200 022", "Other")
S=S.replace("Car Washes  (ES 0200 0150  08)", "Car wash")
S=S.replace("Office Machinery and Equipment Rental / Leasing (ES 0200 022", "Other")

S=S.replace("Data Processing Hosting and Related Services   (ES 0200 02", "Other")

S=S.replace("Power/Commn Line/Related Structures Construction  (ES 0200 0", "Other")

S=S.replace("Hardware Stores", "Other")

S=S.replace("Software Publishers", "Other")

S=S.replace("Offices of Physicians Mental Health Specialists   (ES 0200", "Other")

S=S.replace("Clothing Accessories Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Construction Mining and Forestry Machinery and Equipment R", "Other")

S=S.replace("New Car Dealers  (ES 0200 0225    08)", "Other")

S=S.replace("Fine Arts Schools   (ES 0200 0225    08)", "Other")

S=S.replace("Lessors of Other Real Estate Property", "Other")

S=S.replace("Automotive Glass Replacement Shops  (ES 0200 0225    08)", "Other")
S=S.replace("Freight Transportation Arrangement", "Other")

S=S.replace("All Other Consumer Goods Rental  (ES 0200 0225    08)", "Other")

S=S.replace("Hobby Toy and Game Stores", "Other")

S=S.replace("Women's Clothing Stores  (ES 0200 0225    08)", "Other")

S=S.replace("General Line Grocery Merchant Wholesalers (ES 0200 0225    0", "Other")

S=S.replace("Computer and Computer Peripheral Equipment and Software Merc", "Other")

S=S.replace("All Other Home Furnishings Stores", "Other")

S=S.replace("Sports and Recreation Instruction  (ES 0200 0225  06)", "Other")

S=S.replace("Musical Groups and Artists   (ES 0200 0225    08)", "Other")

S=S.replace("Other Commercial Equipment Merchant Wholesalers", "Other")

S=S.replace("Used Household and Office Goods Moving", "Other")

S=S.replace("Vending Machine Operators", "Other")

S=S.replace("Motor Vehicle Towing  (ES 0200 0225    08)", "Other")

S=S.replace("Vending Machine Operators", "Other")

S=S.replace("Florists", "Other")
S=S.replace("Child and Youth Services   (ES 0200 0225    08)", "Youth services")
S=S.replace("New Housing Operative Builders", "Other")

S=S.replace("Document Preparation Services", "Other")

S=S.replace("Metal Serv Centers/Othr Metal Merch Whlslrs (ES 0200 0225  0", "Other")

S=S.replace("Computer/ Office Machine Repair /Maintenance (ES 0200 0225", "Other")

S=S.replace("Other Specialized Design Services", "Other")

S=S.replace("Remediation Services  (ES 0200 0225    08)", "Other")

S=S.replace("Couriers", "Other")

S=S.replace("Other Foundation/Structure/ Building Extr Contractors (ES 02", "Other")
S=S.replace("Nonresidential Property Managers", "Other")

S=S.replace("Other Electronic Parts/Equipment Merch Wholesalers (ES 0200", "Other")

S=S.replace("All Other General Merchandise Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Surveying / Mapping (ex Geophysical) Services (ES 0200 0225", "Other")

S=S.replace("Appliance Repair and Maintenance   (ES 0200 0225    08)", "Other")

S=S.replace("Commercial Gravure Printing", "Other")

S=S.replace("Building Inspection Services", "Other")

S=S.replace("Employment Placement Agencies", "Other")
S=S.replace("Footwear and Leather Goods Repair   (ES 0200 0225    08)", "Other")

S=S.replace("Legislative Bodies", "Other")

S=S.replace("Hog and Pig Farming", "Other")

S=S.replace("Hair Nail and Skin Care Services", "Other")

S=S.replace("All Other Pipeline Transportation", "Other")

S=S.replace("Truck Trailer Manufacturing", "Other")

S=S.replace("Travel Trailer and Camper Manufacturing", "Other")
S=S.replace("Short Line Railroads  (ES 0200 0225    06)", "Railroads")
S=S.replace("Direct Selling Establishments", "Other")

S=S.replace("Other Support Activities for Air Transportation (ES 0200 022", "Other")

S=S.replace("Unlaminated Plastics Profile Shape Manufacturing (ES 0200 02", "Other")

S=S.replace("Correctional Institutions", "Other")

S=S.replace("Cutting Tool and Machine Tool Accessory Manufacturing", "Other")

S=S.replace("Residential Mental Retardation Facilities", "Other")

S=S.replace("Nonwoven Fabric Mills", "Other")

S=S.replace("Other Measuring and Controlling Device Manufacturing", "Other")

S=S.replace("Truck Trailer Manufacturing", "Other")
S=S.replace("Services for the Elderly & Persons w/Disabilities (ES 0200 0", "Other")

S=S.replace("Pharmacies and Drug Stores", "Other")

S=S.replace("Veterinary Services (ES 0200 0225    08)", "Other")

S=S.replace("Tire Dealers  (ES 0200 0225    08)", "Other")

S=S.replace("Book Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Professional and Management Development Training", "Other")

S=S.replace("Retail Bakeries  (ES 0600 0400  08)", "Other")

S=S.replace("Wood Kitchen Cabinet and Countertop Manufacturing  (ES 0200", "Other")
S=S.replace("Sporting/Recr Goods/Supplies Merch Whlslrs (ES 0200 0225  08", "Other")

S=S.replace("Advertising Agencies", "Other")

S=S.replace("Othr Electronic/Precision Equip Repair/Maintenance (ES 0200", "Other")

S=S.replace("Food Service Contractors", "Other")

S=S.replace("Passenger Car Leasing", "Other")

S=S.replace("Structural Steel and Precast Concrete Contractors  (ES 0200", "Other")

S=S.replace("Auto/Other Motor Vehicle Merch Wholesalers (ES 0200 0225  08", "Other")

S=S.replace("Real Estate Credit   (ES 0200 0225    08)", "Other")
S=S.replace("Engineered Wood Member (except Truss) Manufacturing", "Other")

S=S.replace("Materials Recovery Facilities", "Other")

S=S.replace("Rolled Steel Shape Manufacturing", "Other")

S=S.replace("Oilseed (except Soybean) Farming", "Other")

S=S.replace("Printing and Writing Paper Merchant Wholesalers", "Other")

S=S.replace("Public Finance Activities", "Other")

S=S.replace("Repossession Services", "Other")
S=S.replace("Executive Offices", "Other")

S=S.replace("All Other Misc Nonmetallic Mineral Product Manuf (ES 0200 02", "Other")

S=S.replace("All Other Miscellaneous Electrical Equipment and Component M ", "Other")

S=S.replace("Electronic Computer Manufacturing", "Other")

S=S.replace("Saw Blade and Handtool Manufacturing", "Other")

S=S.replace("Concrete Pipe Manufacturing  (ES 0200 0225  08)", "Other")

S=S.replace("Rice Farming", "Other")
S=S.replace("Other Construction Material Merchant Wholesalers  (ES 0200 0", "Other")

S=S.replace("All Other Specialty Food Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Testing Laboratories", "Other")

S=S.replace("Lessors of Miniwarehouses/Self-Storage Units   (ES 0200 0225", "Other")

S=S.replace("Recyclable Material Merchant Wholesalers  (ES 0200 0225    0", "Other")

S=S.replace("Reupholstery and Furniture Repair   (ES 0200 0225    08)", "Other")

S=S.replace("Commercial Screen Printing  (ES 0200 0225  08)", "Other")

S=S.replace("Financial Transactions Processing Reserve and Clearinghous", "Other")
S=S.replace("Plumbing and Heating Equipment and Supplies (Hydronics) Merc", "Other")

S=S.replace("Coin-Operated Laundries and Drycleaners  (ES 0200 0150  08)", "Other")

S=S.replace("Truck Utility Trailer and RV (Recreational Vehicle) Rental", "Other")

S=S.replace("Other Financial Vehicles", "Other")

S=S.replace("All Other Telecommunications", "Other")

S=S.replace("Drycleaning/Laundry Srvcs (ex Coin-Operated) (ES 0600 0400", "Other")

S=S.replace("Warm Air Heating and Air-Conditioning Equipment and Supplies", "Other")
S=S.replace("Tobacco Stores  (ES 0200 0225    08)", "Tabacco")
S=S.replace("Offices of Optometrists", "Other")

S=S.replace("Food (Health) Supplement Stores", "Other")

S=S.replace("Pet and Pet Supplies Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Investment Banking and Securities Dealing", "Other")

S=S.replace("Industrial and Personal Service Paper Merchant Wholesalers", "Other")

S=S.replace("Mortgage and Nonmortgage Loan Brokers   (ES 0200 0225    08)", "Other")
S=S.replace("Other Social Advocacy Organizations", "Other")

S=S.replace("Sales Financing", "Other")

S=S.replace("Commercial Banking   (ES 0200 0225    08)", "Other")

S=S.replace("Petroleum/ Petroleum Prod Merch Wholesalers (except Bulk Sta", "Other")

S=S.replace("Locksmiths   (ES 0200 0225    08)", "Other")

S=S.replace("Nursery Garden Center and Farm Supply Stores  (ES 0200 022", "Other")
S=S.replace("Septic Tank and Related Services", "Other")

S=S.replace("Other Similar Organizations (except Business Professional", "Other")

S=S.replace("Industrial Supplies Merchant Wholesalers", "Other")

S=S.replace("Construction and Mining (except Oil Well) Machinery and Equi", "Other")

S=S.replace("Industrial Building Construction  (ES  0200 0225   08)", "Other")

S=S.replace("Research/Dvlp in the Social Sciences/Humanities (ES 0200 022", "Other")

S=S.replace("Household Appliance Stores", "Other")

S=S.replace("Other Profess Equip/Supplies Merch Whlslrs (ES 0200 0225  08", "Other")
S=S.replace("Coal and Other Mineral and Ore Merchant Wholesalers", "Other")

S=S.replace("Blood and Organ Banks", "Other")

S=S.replace("Coastal and Great Lakes Freight Transportation  (ES 0200 022", "Other")

S=S.replace("Apiculture", "Other")

S=S.replace("Chocolate and Confectionery Manufacturing from Cacao Beans", "Other")

S=S.replace("Nonferrous Metal (except Copper and Aluminum) Rolling Drawi", "Other")

S=S.replace("Administration of Conservation Programs  (ES 0200 022 5 01)", "Other")
S=S.replace("Search Detection Navigation Guidance Aeronautical and Na", "Other")

S=S.replace("Musical Instrument Manufacturing", "Other")

S=S.replace("Other Motor Vehicle Parts Manufacturing", "Other")

S=S.replace("Television Broadcasting  (ES 0200 0225  06)", "Other")

S=S.replace("Dimension Stone Mining and Quarrying", "Other")

S=S.replace("Hazardous Waste Collection", "Other")

S=S.replace("Postal Service", "Other")
S=S.replace("Business Associations", "Other")

S=S.replace("Internet Publishing and Broadcasting and Web Search Portals", "Other")

S=S.replace("All Other Miscell Ambulatory Health Care Servcs (ES 0200 022", "Other")

S=S.replace("Other Community Housing Services", "Other")

S=S.replace("Shoe Stores", "Other")

S=S.replace("Solid Waste Collection   (ES 0200 0225    08)", "Other")

S=S.replace("Mtr Vehicle Spplies/New Parts Merch Whlslrs (ES 0200 0225  0", "Other")
S=S.replace("Office Supplies and Stationery Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Office Equipment Merchant Wholesalers", "Other")

S=S.replace("Travel Agencies", "Other")

S=S.replace("Credit Unions   (ES 0200 0225    08)", "Other")

S=S.replace("Offices of Real Estate Appraisers   (ES 0200 0225    08)", "Other")

S=S.replace("Roofing/Siding/Insulation Material Merch Whlslrs (ES 0200 02", "Other")

S=S.replace("Exam Preparation and Tutoring", "Other")
S=S.replace("Other Technical and Trade Schools  (ES 0200 0225  06)", "Other")

S=S.replace("Fish and Seafood Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Investigation Services   (ES 0200 0225    08)", "Other")

S=S.replace("Fresh Fruit and Vegetable Merchant Wholesalers (ES 0200 0225", "Other")

S=S.replace("Serv Establishment Equip/Supplies Merch Whlslrs (ES 0200 022", "Other")

S=S.replace("New Multifamily Housing Construction (except Operative Build", "Other")

S=S.replace("Electrical and Electronic Appliance Television and Radio S", "Other")
S=S.replace("Travel Agencies", "Other")

S=S.replace("Other Services Related to Advertising", "Other")

S=S.replace("Jewelry and Silverware Manufacturing", "Other")

S=S.replace("Nursing Care Facilities (ES 0200 0150  06)", "Other")

S=S.replace("Window Treatment Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Grantmaking Foundations", "Other")

S=S.replace("Petroleum Bulk Stations and Terminals  (ES 0200 0225    08)", "Other")

S=S.replace("Dental Laboratories   (ES 0200 0225    06)", "Other")
S=S.replace("Unlaminated Plastics Film and Sheet (except Packaging) Manuf", "Other")

S=S.replace("Asphalt Paving Mixture and Block Manufacturing (ES 0200 0225", "Other")

S=S.replace("Iron and Steel Mills and Ferroalloy Manufacturing", "Other")

S=S.replace("Turbine and Turbine Generator Set Units Manufacturing", "Other")

S=S.replace("Corrugated and Solid Fiber Box Manufacturing (ES 0200 0225", "Other")

S=S.replace("Industrial Gas Manufacturing", "Other")

S=S.replace("Natural Gas Distribution (ES 0200 0225  08)", "Other")

S=S.replace("Soft Drink Manufacturing  (ES 0200 0225  08)", "Other")

S=S.replace("Small Arms Manufacturing", "Other")

S=S.replace("Folding Paperboard Box Manufacturing", "Other")

S=S.replace("Offices of Bank Holding Companies", "Other")

S=S.replace("All Other Miscellaneous Electrical Equipment and Component M", "Other")

S=S.replace("Administration of Public Health Programs", "Other")

S=S.replace("Polish and Other Sanitation Good Manufacturing (ES 0200 0225", "Other")

S=S.replace("Miscellaneous Intermediation", "Other")
S=S.replace("Bed-and-Breakfast Inns", "Hotel")
S=S.replace("Parking Lots and Garages   (ES 0200 0225    08)", "Other")

S=S.replace("Barber Shops  (ES 0200 0225    08)", "Other")

S=S.replace("Iron and Steel Pipe and Tube Manufacturing from Purchased St", "Other")

S=S.replace("Forest Nurseries and Gathering of Forest Products", "Other")

S=S.replace("Other Snack Food Manufacturing", "Other")
S=S.replace("Breweries  (ES 0200 0225  08)", "Other")

S=S.replace("Musical Instrument and Supplies Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Consumer Electronics Repair and Maintenance   (ES 0200 0225", "Other")

S=S.replace("Other Grantmaking and Giving Services", "Other")

S=S.replace("Process Physical Distribution and Logistics Consulting Ser", "Other")

S=S.replace("General Warehousing and Storage  (ES 0200 0225    08)", "Other")

S=S.replace("Passenger Car Rental   (ES 0200 0225    08)", "Other")

S=S.replace("Newspaper Publishers", "Other")
S=S.replace("Court Reporting and Stenotype Services", "Other")

S=S.replace("Support Activities for Rail Transportation   (ES 0200 0225", "Other")

S=S.replace("Other Residential Care Facilities  (ES 0200 0225 06)", "Other")

S=S.replace("Commercial Bakeries  (ES 0800 0400    08)", "Other")

S=S.replace("All Other Automotive Repair and Maintenance", "Other")

S=S.replace("Home Furnishing Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("All Other Information Services", "Other")

S=S.replace("All Other Plastics Product Manufacturing  (ES 0200 0225  08)", "Other")
S=S.replace("Other Performing Arts Companies", "Other")

S=S.replace("Hardware Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Research and Development in Biotechnology", "Other")

S=S.replace("Marine Cargo Handling  (ES 0200 0225    08)", "Other")

S=S.replace("Medical Laboratories   (ES 0200 0225    08)", "Other")

S=S.replace("Fabricated Structural Metal Manufacturing (ES 0200 0225    0", "Other")

S=S.replace("All Other Miscellaneous Wood Product Manufacturing", "Other")
S=S.replace("All Other Miscellaneous Textile Product Mills", "Other")

S=S.replace("Wine and Distilled Alcoholic Beverage Merchant Whlslrs (0200", "Other")

S=S.replace("Jewelry Watch Precious Stone and Precious Metal Merchant", "Other")

S=S.replace("Sewing Needlework and Piece Goods Stores", "Other")

S=S.replace("All Other Miscellaneous Fabricated Metal Product Manufacturi", "Other")

S=S.replace("Mail-Order Houses", "Other")

S=S.replace("Machine Shops", "Other")

S=S.replace("Convention and Trade Show Organizers  (ES 0200 0225    08)", "Other")
S=S.replace("All Other Insurance Related Activities", "Other")

S=S.replace("Baked Goods Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Nursery and Tree Production", "Other")

S=S.replace("Cut and Sew Apparel Contractors", "Other")

S=S.replace("Other Waste Collection", "Other")

S=S.replace("All Other Nondepository Credit Intermediation   (ES 0200 022", "Other")

S=S.replace("Fuel Dealers", "Other")

S=S.replace("Brick/Stone/Related Const Material Merch Whlslrs  (ES 0200 0", "Other")
S=S.replace("Concrete Block and Brick Manufacturing", "Other")

S=S.replace("Pipeline Transportation of Natural Gas  (ES 0200 0225    08)", "Other")

S=S.replace("Mixed Mode Transit Systems  (ES 0200 0225    08)", "Other")

S=S.replace("Commodity Contracts Dealing", "Other")

S=S.replace("Solar Electric Power Generation", "Other")

S=S.replace("Dog and Cat Food Manufacturing", "Other")

S=S.replace("Lawn and Garden Tractor and Home Lawn and Garden Equipment M", "Other")

S=S.replace("Finfish Fishing", "Other")

S=S.replace("Administration of Housing Programs", "Other")

S=S.replace("School and Employee Bus Transportation", "Other")

S=S.replace("Floriculture Production", "Other")

S=S.replace("Legal Counsel and Prosecution", "Other")

S=S.replace("Home Health Equipment Rental  (ES 0200 0225    08)", "Other")

S=S.replace("Paper Bag and Coated and Treated Paper Manufacturing", "Other")
S=S.replace("Metal Window and Door Manufacturing", "Other")

S=S.replace("Industrial Truck Tractor Trailer and Stacker Machinery Ma", "Other")

S=S.replace("Dairy Cattle and Milk Production", "Other")

S=S.replace("Flat Glass Manufacturing", "Other")

S=S.replace("Food Product Machinery Manufacturing", "Other")

S=S.replace("Switchgear and Switchboard Apparatus Manufacturing", "Other")

S=S.replace("Soil Preparation Planting and Cultivating", "Other")

S=S.replace("Footwear Manufacturing", "Other")

S=S.replace("Offices of Notaries", "Other")

S=S.replace("Other Pressed /Blown Glass/Glassware Manufacturing (ES 0200", "Other")

S=S.replace("Motor and Generator Manufacturing", "Other")

S=S.replace("Hardware Manufacturing", "Other")

S=S.replace("Iron Foundries  (ES 0200 0225    08)", "Other")
S=S.replace("Specialty (except Psychiatric and Substance Abuse) Hospitals", "Hospital")
S=S.replace("Sheet Metal Work Manufacturing  (ES 0200 0225    08)", "Other")

S=S.replace("Wood Container and Pallet Manufacturing (ES 0200 0225  08)", "Other")

S=S.replace("Consumer Lending   (ES 0200 0225    08)", "Other")

S=S.replace("Optical Goods Stores", "Other")

S=S.replace("Collection Agencies", "Other")

S=S.replace("All Other Miscellaneous Waste Management Services (ES 0200 0", "Other")

S=S.replace("Other Cut and Sew Apparel Manufacturing", "Other")

S=S.replace("Warehouse Clubs and Supercenters  (ES 0200 0225    08)", "Other")
S=S.replace("Land Subdivision  (ES 0200 0225  08)", "Other")

S=S.replace("Dairy Product (ex Dried or Canned) Merch Whlslrs (ES 0200 02", "Other")

S=S.replace("Limousine Service (ES 0200 0225    08)", "Other")

S=S.replace("Securities Brokerage", "Other")

S=S.replace("Nonupholstered Wood Household Furniture Manuf (ES 0200 0225", "Other")

S=S.replace("Sound Recording Studios  (ES 0200 0225    08)", "Other")

S=S.replace("Other Gasoline Stations  (ES 0200 0225    08)", "Other")
S=S.replace("Packaged Frozen Food Merchant Wholesalers  (ES 0200 0225", "Other")

S=S.replace("Flower/Nursery Stock/ Florists' Supplies MerchWhlslrs (ES 02", "Other")

S=S.replace("All Other Support Activities for Transportation   (ES 0200 0", "Other")

S=S.replace("Fruit and Vegetable Markets", "Other")

S=S.replace("Ornamental/Architectural Metal Work Manuf (ES 0200 0225  08)", "Other")

S=S.replace("Drafting Services", "Other")
S=S.replace("Special Needs Transportation", "Other")

S=S.replace("Furniture Merchant Wholesalers", "Other")

S=S.replace("Oil and Gas Pipeline and Related Structures Construction", "Other")

S=S.replace("Other Warehousing and Storage   (ES 0200 0225    08)", "Other")

S=S.replace("Voluntary Health Organizations   (ES 0200 0225    08)", "Other")

S=S.replace("Direct Property and Casualty Insurance Carriers   (ES 0200 0", "Other")

S=S.replace("Title Abstract and Settlement Offices   (ES 0200 0225    08)", "Other")
S=S.replace("Sports Teams and Clubs  (ES 0200 0225    08)", "Other")

S=S.replace("Other Food Crops Grown Under Cover", "Other")

S=S.replace("Funeral Homes and Funeral Services  (ES 0800 0700  08)", "Other")

S=S.replace("Corporate Subsidiary and Regional Managing Offices", "Other")

S=S.replace("Petroleum Refineries", "Other")

S=S.replace("All Other Miscellaneous General Purpose Machinery Manufactur", "Other")

S=S.replace("Research and Development in the Physical Engineering and L", "Other")
S=S.replace("Portfolio Management   (ES 0200 0225    08)", "Other")

S=S.replace("General Rental Centers  (ES 0200 0225    08)", "Other")

S=S.replace("Advertising Material Distribution Services", "Other")

S=S.replace("Other Farm Product Raw Material Merch Whlslrs (ES 0200 0225", "Other")

S=S.replace("All Other Miscellaneous Food Manufacturing  (ES 0200 0225  0", "Other")

S=S.replace("Confectionery Merchant Wholesalers", "Other")

S=S.replace("Integrated Record Production/Distribution", "Other")
S=S.replace("Frozen Fruit Juice and Vegetable Manufacturing", "Other")

S=S.replace("Scheduled Freight Air Transportation", "Other")

S=S.replace("Blind and Shade Manufacturing", "Other")

S=S.replace("Animal (except Poultry) Slaughtering (ES 0200 0225  08)", "Other")

S=S.replace("Telephone Apparatus Manufacturing", "Other")

S=S.replace("National Security", "Other")

S=S.replace("Tour Operators   (ES 0200 0225    08)", "Other")

S=S.replace("Chicken Egg Production", "Other")

S=S.replace("Rubber and Plastics Hoses and Belting Manufacturing", "Other")

S=S.replace("Pharmaceutical Preparation Manufacturing", "Other")

S=S.replace("Other Spectator Sports", "Other")

S=S.replace("Business to Business Electronic Markets", "Other")

S=S.replace("Solid Waste Landfill", "Other")

S=S.replace("Welding and Soldering Equipment Manufacturing", "Other")

S=S.replace("Inland Water Freight Transportation", "Other")
S=S.replace("Administration of General Economic Programs", "Other")

S=S.replace("Sporting and Athletic Goods Manufacturing", "Other")

S=S.replace("Executive Search Services", "Other")

S=S.replace("Other Lighting Equipment Manufacturing", "Other")

S=S.replace("Railroad Rolling Stock Manufacturing  (ES 0200 0225  08)", "Other")

S=S.replace("Administration of Air and Water Resource and Solid Waste Man", "Other")

S=S.replace("Ready-Mix Concrete Manufacturing  (ES 0200 0225  08)", "Other")

S=S.replace("Direct Title Insurance Carriers   (ES 0200 0225    08)", "Other")

S=S.replace("Support Activities for Oil and Gas Operations", "Other")

S=S.replace("Fruit and Vegetable Canning (ES 0200 0225  08)", "Other")

S=S.replace("Port and Harbor Operations  (ES 0200 0225    08)", "Other")

S=S.replace("Plastics Pipe and Pipe Fitting Manufacturing  (ES 0200 0225", "Other")

S=S.replace("Other Animal Food Manufacturing", "Other")

S=S.replace("Machine Tool Manufacturing", "Other")
S=S.replace("Home/Garden Equip Repair /Maintenance (ES 0200 0225  08)", "Other")

S=S.replace("Transp Equip(ex Mtr Vhcle) Merch Whlslrs (ES 0200 0225  08)", "Other")

S=S.replace("Linen Supply (ES 0400 0300   08)l", "Other")

S=S.replace("Surgical and Medical Instrument Manufacturing", "Other")

S=S.replace("Boat Building  (ES 0200 0225  08)", "Other")

S=S.replace("HMO Medical Centers", "Other")

S=S.replace("Glass Product Manufacturing Made of Purchased Glass (ES 0200", "Other")
S=S.replace("All Other Traveler Accommodation", "Other")

S=S.replace("Water Supply and Irrigation Systems", "Other")

S=S.replace("All Other Travel Arrangement and Reservation Services", "Other")

S=S.replace("Public Relations Agencies", "Other")

S=S.replace("Men's Clothing Stores  (ES 0200 0225    08)", "Other")

S=S.replace("HMO Medical Centers", "Other")

S=S.replace("Other Communications Equipment Manufacturing", "Other")
S=S.replace("Toy/Hobby Goods/Supplies Merch Wholesalers (ES 0200 0225  08", "Other")

S=S.replace("Agents and Managers for Artists Athletes Entertainers and", "Other")

S=S.replace("Motor Vehicle Parts (Used) Merchant Wholesalers", "Other")

S=S.replace("Computer Training   (ES 0200 0225    08)", "Other")

S=S.replace("Linen Supply (ES 0400 0300   08)", "Other")

S=S.replace("Family Planning Centers", "Other")
S=S.replace("Tire and Tube Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Consumer Electronics and Appliances Rental (ES 0200 0225", "Other")

S=S.replace("Human Resources and Executive Search Consulting Services", "Other")

S=S.replace("Professional Employer Organizations", "Other")

S=S.replace("Diagnostic Imaging Centers", "Other")

S=S.replace("Ship Building and Repairing  (ES 0200 0225  08)", "Other")

S=S.replace("Pump and Pumping Equipment Manufacturing", "Other")
S=S.replace("Other Auto Mech/Electrical Repair/Maintenance (ES 0200 0225", "Other")

S=S.replace("Other Commercial and Service Industry Machinery Manufacturin", "Other")

S=S.replace("All Other Misc Chemical Product/Preparation Manuf (ES 0200 0", "Other")

S=S.replace("Stationery and Office Supplies Merchant Wholesalers", "Other")

S=S.replace("Other Business Srv Cntrs (including Copy Shops) (ES 0200 022", "Other")

S=S.replace("Other Nonhazardous Waste Treatment and Disposal   (ES 0200 0", "Other")

S=S.replace("Payroll Services", "Other")

S=S.replace("Periodical Publishers", "Other")

S=S.replace("Satellite Telecommunications", "Other")
S=S.replace("Professional Organizations", "Other")

S=S.replace("Communication Equipment Repair and Maintenance", "Other")

S=S.replace("Motorcycle ATV and All Other Motor Vehicle Dealers", "Other")

S=S.replace("Farm Supplies Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Apprenticeship Training", "Other")

S=S.replace("Power/Distrib/Specialty Transformer Manuf (ES 0200 0225  08)", "Other")

S=S.replace("Other Concrete Product Manufacturing (ES 0200 0225  08)", "Other")

S=S.replace("Other Sound Recording Industries", "Other")

S=S.replace("Parking Lots and Garages   (ES 0200 0225    08)", "Other")

S=S.replace("Barber Shops  (ES 0200 0225    08)", "Other")

S=S.replace("Iron and Steel Pipe and Tube Manufacturing from Purchased St", "Other")

S=S.replace("Forest Nurseries and Gathering of Forest Products", "Other")

S=S.replace("Other Snack Food Manufacturing", "Other")
S=S.replace("Museums  (ES 0200 0225  06)", "Museum")

S=S.replace("Cosmetology and Barber Schools   (ES 0200 0225    08)", "Cosmetology and Barber Schools")

S=S.replace("Amusement Arcades", "Amusement")
S=S.replace("Paint and Wallpaper Stores  (ES 0200 0225    08)", "Other")

S=S.replace("Plastics Material and Resin Manufacturing", "Other")

S=S.replace("Admin of Human Resource Programs (expt Education Public Hea", "Other")

S=S.replace("Soap and Other Detergent Manufacturing", "Other")

S=S.replace("General Freight Trucking Long-Distance Less Than Truckload", "Other")
S=S.replace("Community Food Services", "Soup kitchens")

S=S.replace("General Medical and Surgical Hospitals   (ES 0200 0225    08", "Hospital")

S=S.replace("Temporary Shelters  (ES 0200 0225  06)", "Shelters")
S=S.replace("General Freight Trucking Long-Distance Less Than Truckload", "Other")

S=S.replace("Residential Mental Health and Substance Abuse Facilities", "Other")

S=S.replace("Hazardous Waste Treatment and Disposal", "Other")

S=S.replace("Footwear Merchant Wholesalers", "Other")

S=S.replace("Pottery Ceramics and Plumbing Fixture Manufacturing", "Other")

S=S.replace("Wood Window and Door Manufacturing (ES 0200 0225  08)", "Other")
S=S.replace("Home Centers  (ES 0200 0225    08)", "Other")

S=S.replace("Soap and Other Detergent Manufacturing", "Other")

S=S.replace("Book Periodical and Newspaper Merchant Wholesalers", "Other")

S=S.replace("Paint Varnish and Supplies Merchant Wholesalers  (ES 0200", "Other")

S=S.replace("Plastics Materials/Basic Forms/Shapes Merch Whlslrs (ES 0200", "Other")

S=S.replace("Facilities Support Services", "Other")
S=S.replace("Environment Conservation and Wildlife Organizations", "Other")

S=S.replace("Perishable Prepared Food Manufacturing  (ES 0200 0225  08)", "Other")

S=S.replace("Radio&TV Broadcast & Wireless Comm Equipment Manuf (ES 0200", "Other")

S=S.replace("Other Activities Related to Credit Intermediation (ES 0200 0", "Other")

S=S.replace("Other Industrial Machinery Manufacturing", "Other")

S=S.replace("Toilet Preparation Manufacturing", "Other")

S=S.replace("Meat Markets  (ES 0200 0225    08)", "Other")
S=S.replace("Poultry and Poultry Product Merchant Wholesalers", "Other")

S=S.replace("Cookie and Cracker Manufacturing", "Other")

S=S.replace("All Other Transportation Equipment Manufacturing", "Other")

S=S.replace("Miscellaneous Financial Investment Activities", "Other")

S=S.replace("Stationery Product Manufacturing", "Other")

S=S.replace("Emergency and Other Relief Services", "Other")

S=S.replace("Outdoor Power Equipment Stores  (ES 0200 0225    04)", "Other")

S=S.replace("Department Stores (except Discount Department Stores)", "Other")
S=S.replace("Line-Haul Railroads (ES 0200 0225    06)", "Railroads")

S=S.replace("Cafeterias Grill Buffets and Buffets", "Restaurant")

S=S.replace("Bowling Centers  (ES 0400 0300  08)", "Bowling")

S=S.replace("Libraries and Archives  (ES 0200 0225  06)", "Libraries")
S=S.replace("Other Justice Public Order and Safety Activities", "Other")

S=S.replace("Clay Building Material and Refractories Manufacturing", "Other")

S=S.replace("Telephone Answering Services", "Other")

S=S.replace("Administration of Education Programs", "Other")

S=S.replace("Savings Institutions   (ES 0200 0225    08)", "Other")

S=S.replace("Instruments & Related Prods Manufacturing to Measure Displa", "Other")

S=S.replace("Direct Life Insurance Carriers   (ES 0200 0225    08)", "Other")
S=S.replace("Bolt Nut Screw Rivet and Washer Manufacturing", "Other")

S=S.replace("All Other Converted Paper Product Manufacturing", "Other")

S=S.replace("Rendering and Meat Byproduct Processing (ES 0200 0225  08)", "Other")

S=S.replace("Teleproduction and Other Postproduction Services", "Other")

S=S.replace("Refrigeration Equip/Supplies Merchant Whlslrs (ES 0200 0225", "Other")

S=S.replace("Commercial Industrial and Institutional Electric Lighting", "Other")

S=S.replace("Freestanding Ambulatory Surgical and Emergency Centers", "Hospital")

S=S.replace("Distilleries", "Other")

S=S.replace("Children's and Infants' Clothing Stores  (ES 0200 0225    08", "Other")

S=S.replace("Rolling Mill and Other Metalworking Machinery Manufacturing", "Other")
S=S.replace("Metal Coating Engraving (exJewelry and Silverware) and All", "Other")

S=S.replace("Fish and Seafood Markets  (ES 0200 0225    08)s", "Other")

S=S.replace("Radio Networks", "Other")

S=S.replace("Truss Manufacturing", "Other")

S=S.replace("Recreational Vehicle Dealers", "Other")

S=S.replace("Diet and Weight Reducing Centers", "Other")

S=S.replace("All Other Miscellaneous Crop Farming", "Other")

S=S.replace("Administration of Urban Planning and Community and Rural Dev", "Other")

S=S.replace("Spice and Extract Manufacturing", "Other")

S=S.replace("Instrument Manufacturing for Measuring and Testing Electrici", "Other")

S=S.replace("Kidney Dialysis Centers  (ES 0200 0225    08)", "Other")

S=S.replace("Offices of Podiatrists", "Other")

S=S.replace("Trust Fiduciary and Custody Activities", "Other")
S=S.replace("Electronic Auctions", "Other")

S=S.replace("Textile Bag and Canvas Mills", "Other")

S=S.replace("Men's and Boys' Clothing and Furnishings Merchant Wholesaler", "Other")

S=S.replace("Prefabricated Metal Building/Component Manuf (ES 0200 0225", "Other")

S=S.replace("Refrigerated Warehousing and Storage  (ES 0200 0225    08)", "Other")

S=S.replace("Other Millwork (including Flooring)", "Other")

S=S.replace("Photographic Equipment and Supplies Merchant Wholesalers", "Other")

S=S.replace("Coffee and Tea Manufacturing", "Other")

S=S.replace("Telemarketing Bureaus", "Other")

S=S.replace("Institutional Furniture Manufacturing", "Other")

S=S.replace("Video Tape and Disc Rental", "Other")
S=S.replace("Theater Companies and Dinner Theaters   (ES 0200 0225    08)", "Theater")

S=S.replace("Tobacco and Tobacco Product Merchant Wholesalers", "Tabacco")
S=S.replace("Direct Health and Medical Insurance Carriers  (ES 0200 0225", "Other")

S=S.replace("Automobile Driving Schools   (ES 0200 0225    08)", "Other")

S=S.replace("Heating Equipment (except Warm Air Furnaces) Manufacturing", "Other")

S=S.replace("Scenic and Sightseeing Transportation Water", "Other")

S=S.replace("Wood Office Furniture Manufacturing", "Other")

S=S.replace("Labor Unions and Similar Labor Organizations", "Other")

S=S.replace("Armored Car Services   (ES 0200 0225    08)", "Other")

S=S.replace("Fire Protection", "Other")

S=S.replace("Scenic and Sightseeing Transportation Land", "Other")

S=S.replace("Book Publishers", "Other")
S=S.replace("Cable and Other Subscription Programming  (ES 0200 0225  06)", "Other")

S=S.replace("Audio and Video Equipment Manufacturing", "Other")

S=S.replace("Other General Government Support", "Other")

S=S.replace("Language Schools", "Other")

S=S.replace("Navigational Services to Shipping", "Other")

S=S.replace("Logging", "Other")
S=S.replace("All Other Outpatient Care Centers   (ES 0200 0225    08)", "Other")

S=S.replace("Other Motion Picture and Video Industries", "Other")

S=S.replace("Ophthalmic Goods Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Support Activities for Animal Production   (ES 0200 0225", "Other")

S=S.replace("Lessors of Nonfinancial Intangible Assets (except Copyrighte", "Other")

S=S.replace("All Other Basic Organic Chemical Manufacturing  (ES 0200 022", "Other")

S=S.replace("Sawmills   (ES 0200 0225    06)", "Other")

S=S.replace("Industrial Launderers   (ES 0600 0400  08)", "Other")

S=S.replace("Curtain and Linen Mills", "Other")

S=S.replace("Confectionery and Nut Stores  (ES 0200 0225    08)", "Other")
S=S.replace("Private Households", "Other")

S=S.replace("Geophysical Surveying and Mapping Services", "Other")

S=S.replace("Commercial Air Rail and Water Transportation Equipment Ren", "Other")

S=S.replace("Showcase Partition Shelving and Locker Manufacturing", "Other")

S=S.replace("Claims Adjusting", "Other")
S=S.replace("Rooming and Boarding Houses  (ES 0200 0225  06)", "Hotel")
S=S.replace("Beer and Ale Merchant Wholesalers  (ES 0200 0225    08)", "Other")

S=S.replace("Other Direct Insuran (ex Life/Health/ Med) Carriers (ES 0200", "Other")

S=S.replace("Electromedical and Electrotherapeutic Apparatus Manufacturin", "Other")

S=S.replace("Other Metal Container Manufacturing  (ES 0200 0225    08)", "Other")

S=S.replace("Confectionery Manufacturing from Purchased Chocolate", "Other")

S=S.replace("All Other Animal Production   (ES 0200 0225    01)", "Other")

S=S.replace("Prefabricated Wood Building Manufacturing", "Other")

S=S.replace("Other Support Activities for Water Transportation", "Other")

S=S.replace("Recreational Goods Rental", "Other")
S=S.replace("Analytical Laboratory Instrument Manufacturing", "Other")

S=S.replace("Bottled Water Manufacturing", "Other")

S=S.replace("Cemeteries and Crematories", "Other")

S=S.replace("Continuing Care Retirement Communities", "Other")

S=S.replace("Ice Cream and Frozen Dessert Manufacturing", "Other")

S=S.replace("Ambulance Services", "Other")

S=S.replace("Apparel Accessories and Other Apparel Manufacturing", "Other")

S=S.replace("Third Party Administration of Insurance and Pension Funds", "Other")

S=S.replace("All Other Publishers", "Other")

S=S.replace("Credit Bureaus   (ES 0200 0225    08)", "Other")
S=S.replace("Air-Con/Warm Air Heat Eq/Com/Indust Refrig Eq Manuf (ES 0200", "Other")

S=S.replace("Construction Sand and Gravel Mining   (ES 0200 0225    04)", "Other")

S=S.replace("Totalizing Fluid Meter and Counting Device Manufacturing", "Other")

S=S.replace("Support Activites for Printing", "Other")

S=S.replace("Support Activities for Forestry", "Other")

S=S.replace("Cut Stone and Stone Product Manufacturing  (ES 0200 0225", "Other")

S=S.replace("Marketing Research and Public Opinion Polling", "Other")

S=S.replace("Record Production", "Other")
S=S.replace("Motion Picture Theaters (except Drive-Ins)  (ES 0200 0225", "Other")

S=S.replace("Display Advertising", "Other")

S=S.replace("Software and Other Prerecorded Compact Disc Tape and Recor", "Other")

S=S.replace("Farm and Garden Machinery and Equipment Merchant Wholesalers", "Other")

S=S.replace("Human Rights Organizations", "Other")

S=S.replace("Women'/ Children's / Infants' Clothing and Accessories Merch", "Other")

S=S.replace("Computer Facilities Management Services", "Other")

S=S.replace("Seafood Product Preparation and Packaging", "Other")
S=S.replace("Women's Girls' and Infants' Cut and Sew Apparel Manufactur", "Other")

S=S.replace("Mayonnaise Dressing and Other Prepared Sauce Manufacturing", "Other")

S=S.replace("Surgical Appliance and Supplies Manufacturing  (ES 0200 0225", "Other")

S=S.replace("Cut Stock Resawing Lumber and Planing   (ES 0200 0225   06", "Other")

S=S.replace("Paint and Coating Manufacturing (ES 0500 0450    08)", "Other")

S=S.replace("Industrial Design Services", "Other")

S=S.replace("Steel Foundries (except Investment) (ES 0200 0225    08)", "Other")

S=S.replace("Piece Goods Notions and Other Dry Goods Merchant Wholesale", "Other")
S=S.replace("Dance Companies", "Other")

S=S.replace("Automotive Transmission Repair   (ES 0200 0225    08)", "Other")

S=S.replace("Custom Architectural Woodwork and Millwork Manufacturing", "Other")

S=S.replace("Poultry Processing", "Other")

S=S.replace("Fish and Seafood Markets  (ES 0200 0225    08)", "Other")

S=S.replace("Sawmill Woodworking and Paper Machinery Manufacturing", "Other")
S=S.replace("Elevator and Moving Stairway Manufacturing  (ES 0200 0225", "Other")

S=S.replace("Golf Courses and Country Clubs  (ES 0200 0225  06)", "Other")

S=S.replace("Luggage and Leather Goods Stores", "Other")

S=S.replace("Discount Department Stores", "Other")

S=S.replace("Packaging and Labeling Services", "Other")

S=S.replace("Trusts Estates and Agency Accounts", "Other")
#finish grouping

S=S.replace("Bowling", "Amusement")

S=S.replace("Theater", "Amusement")

S=S.replace("Museum", "Amusement")
S=S.replace("Convention and Visitors Bureaus", "Other")

S=S.replace("Car wash", "Other")

S=S.replace("Cosmetology and Barber Schools", "Other")

S=S.replace("Shelters", "Other")

S=S.replace("Soup kitchens", "Other")
S=S.replace("Offices of Other Holding Companies", "Other")

S=S.replace("Vocational Rehabilitation Services", "Other")

S=S.replace("Meat and Meat Product Merchant Wholesalers", "Other")

S=S.replace("Specialized Freight (except Used Goods) Trucking Long-Dista", "Other")

S=S.replace("Boat Dealers  (ES 0200 0225    08)", "Other")

S=S.replace("Marinas", "Other")

S=S.replace("Discount Department Stores", "Other")
S['NAICS CODE DESCRIPTION'].value_counts()
S.rename(columns = {'NAICS CODE DESCRIPTION':'Business Type'}, inplace = True)
#Label encode the column

from sklearn.preprocessing import LabelEncoder

S['Business Type']=S['Business Type'].astype('category')

S['Business Types']=S['Business Type'].cat.codes
#Join the edited column back to df

K=Bizns.join(S)

K.head(3)
#Get numerical data

K_Mean=K[['CITY', 'Business Types' ]]

K_Mean.head(3)
#If the city is Tacoma it changes to 1, else it is 0

K_Mean.loc[K_Mean.CITY != 'TACOMA', 'CITY']=0

K_Mean.loc[K_Mean.CITY == 'TACOMA', 'CITY']=1
K_Mean['CITY'].value_counts()
K_Mean.dtypes
K_Mean.describe()
K_Mean.corr()
X=K_Mean[['CITY']]

Y=K_Mean[['Business Types']]
import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

#elbow curve is then graphed with a range from 1 to 20 (represents the number of clusters)

#and the score variable denotes the percentage of variance explained by the number of clusters.



Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
pca = PCA(n_components=1).fit(Y)

pca_d = pca.transform(Y)

pca_c = pca.transform(X)
kmeans=KMeans(n_clusters=5)

kmeansoutput=kmeans.fit(Y)

kmeansoutput

pl.figure('5 Cluster K-Means')

pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)

pl.xlabel('Dividend Yield')

pl.ylabel('Returns')

pl.title('5 Cluster K-Means')

pl.show()
Tree=K_Mean[['CITY', 'Business Types']]

Tree.head(2)
#Decision tree for regression



X1 = K_Mean[['Business Types']]

y1 = K_Mean['CITY'] #Target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.4, random_state=0)
#predict

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
preds=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

preds.head(3)
#Evaluate

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#tree classifier

from sklearn import tree



model = tree.DecisionTreeClassifier()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_predict)
t=pd.DataFrame(

    metrics.confusion_matrix(y_test, y_predict),

    columns=['Predicted Not in Tacoma', 'Predicted in Tacoma'],

    index=['Not really in Tacoma', 'Really in Tacoma']

)

t
#Decision Tree Regression

plt.scatter(X1,y1)

plt.plot(X1, regressor.predict(X1))

plt.title('In or Out (Regression)')

plt.xlabel('Business Type')

plt.ylabel('Tacoma')

plt.show
#Visual the model results

plt.scatter(X1,y1)

plt.plot(X1, model.predict(X1))

plt.title('In or Out of Tacoma')

plt.xlabel('Business Type')

plt.ylabel('Tacoma')

plt.show
One=pd.concat([cViolations, Tacoma_Crime, Bizns, TrafikBoom], axis=1)

One.head(2)
Tac1=One[['OPEN DATE','New Georeferenced Column', 'Crime', 'Occurred On', 'Approximate Time', 'BUSINESS NAME', 'CITY','ZIP CODE', 'NAICS CODE DESCRIPTION', 'Location 1', 'DATE', '24 Hr Time', 'TOTAL FATALITIES', 'Year']]

C=Crime[['CrimeCats']]
Tac2=Tac1.join(C)
Bz=BiHnR[['Hit and Run_False', 'Hit and Run_True']]
Tac3=Tac2.join(Bz)
K_Mean.rename(columns={"CITY": "In Tacoma"}, inplace=True)
Tac4=Tac3.join(K_Mean)
Neighbor=Tac4[['Crime', 'CrimeCats', 'Hit and Run_True','NAICS CODE DESCRIPTION', 'Business Types', 'In Tacoma', 'CITY']]

Neighbor.rename(columns = {'NAICS CODE DESCRIPTION':'Business Type', 'Hit and Run_True': 'Hit and Run', 'CITY': 'City'}, inplace = True)
Neighbor=Neighbor.replace(np.NaN, 0)
#Label encode City



Neighbor['City']= Neighbor['City'].astype('category')

Neighbor['CityCats']= Neighbor['City'].cat.codes

Neighbor.head(3)
Neighbor.describe()
Neighbor.corr()
#Standardize the variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(Neighbor.drop(['Crime', 'Business Type', 'City'], axis=1))
scaled_features = scaler.transform(Neighbor.drop(['Crime', 'Business Type', 'City'], axis=1))
df_n = pd.DataFrame(scaled_features)

df_n.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,Neighbor['In Tacoma'],

                                                    test_size=0.10)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=41576)

knn.fit(X_train,y_train)
pred = knn.predict(X_test) 
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#Choose K Value

import sys

error_rate = []

# Might take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')