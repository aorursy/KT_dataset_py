import pandas as pd

import re

import random

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/global-shark-attacks/GSAF5.csv', encoding="latin-1")

data.head()
data.shape
data.columns
data.dtypes
null_cols = data.isnull().sum()

null_cols[null_cols > 0]
data['Unnamed: 22'].fillna(0, inplace=True)

[x for x in data['Unnamed: 22'] if x!=0]
data['Unnamed: 23'].fillna(0, inplace=True)

[y for y in data['Unnamed: 23'] if y!=0]
data['Case Number.1'].equals(data['Case Number.2'].equals(data['Case Number']))
data['Case Number.1'].equals(data['Case Number.2'])
data['Case Number.1'].isin(data['Case Number.2']).value_counts()

data['Case Number'].equals(data['Case Number.1'])
data['Case Number'].isin(data['Case Number.1']).value_counts()
data['href formula'].equals(data['href'])
data['href formula'].isin(data['href']).value_counts()
data = data.drop(["Unnamed: 22", "Unnamed: 23","Case Number.1","Case Number.2","Time",

                  "href formula",'pdf','original order'], axis=1)

data.head()
set(data['Date'].sample(n=30))
len(set(data['Date']))
try:

    data['Year_']=[d[0:4] for d in data['Case Number']]

except:

    data['Year_']='00'

try:

    data['Month']=[d[5:7] for d in data['Case Number']]

except:

    data['Month']='00'

try:

    data['Day']=[d[8:10] for d in data['Case Number']]

except:

    data['Day']='00'

data.loc[data['Year_'].str.contains('(?i)ND'),'Year_']='00'

data.loc[data['Year_'].str.contains('ND'),'Day']='00'

data.loc[data['Year_'].str.contains('ND'),'Month']='00'

data['Year_'] = data['Year_'].replace(['0.02','0.03','0.04','0.07'],0)

data['Year_'] = data['Year_'].astype(int)

data = data[['Case Number','Day','Month','Year_','Year','Date','Type','Country','Area','Location','Activity','Name',

           'Sex ','Age','Injury','Fatal (Y/N)','Species ','Investigator or Source','href']]

data.sample(n=10)
data[['Year','Year_']].sample(n=20)
data['test'] = data['Year_']==(data['Year'])

data['test'].value_counts()
data.loc[data['Year_']!=(data['Year'])]
data.loc[4983,'Year']=1923

data.loc[3662,'Year']=1961

data.loc[2449,'Year']=1989
data.rename(columns={'Date': 'Date comment'}, inplace=True)
set(data['Fatal (Y/N)'])
data['Fatal (Y/N)'] = data['Fatal (Y/N)'].str.strip()

data['Fatal (Y/N)'] = data['Fatal (Y/N)'].fillna('U')

data['Fatal (Y/N)'] = data['Fatal (Y/N)'].str.replace('n', 'N')

data['Fatal (Y/N)'] = data['Fatal (Y/N)'].str.replace('#VALUE!', 'U')

data['Fatal (Y/N)'] = data['Fatal (Y/N)'].str.replace('F', 'Y')

data['Fatal (Y/N)'] = data['Fatal (Y/N)'].str.replace('UNKNOWN', 'U')

data.rename(columns={'Fatal (Y/N)': 'Fatal (Y/N/U)'}, inplace=True)

print(set(data['Fatal (Y/N/U)']))
data['Number of victims']=data['Name']

data['Number of victims'].fillna('Unknown', inplace=True)

data.head()


unknown=['boat','All on board perished in the crash','a charter fishing boat with James Whiteside and his party','6 m skiboat, occupants: P.A. Reeder & crew','6 m boat, occupants John Winslet & customers','5.4 m boat','5 m skiboat',"""40' fishing cutter""","40' bonito boat","4 boats","""30' cabin cruiser owned by Stefano Catalani""","25-foot cutter","""25' cutter""","""25' rigid-hulled inflatable boat, HI-2""","""24' yacht owened by  C.L. Whitrow""","""22' pleasure boat""","""20' boat of Frank Stocker""",'2 USAF 4-engine planes (an HC-54 & a HC-97) each with 12 onboard collided in mid-air at low altitude and plunged into the Atlantic Ocean',"2 boats","""18' boat of Morris Vorenberg""","""16' launch""","""16' cabin cruiser with 35 hp outboard motor""","""15' boat""","""14' boat, occupant: Jonathan Leodorn""",'14-foot boat Sintra',"""12' dinghy""","""12' boat of Alf Laundy""",'10m boat Lucky Jim',"""10' dinghy""",'10.7 m boat. Occupants: John Capella & friends','Because of a mistaken belief that there were no survivors and several other successive errors, of the 100 to 150 men who survived the sinking, only 11 were rescued. Four of the Sullivan brothers died in the initial blast. ','boat','boat  Marie',"""Boat “Coca Cola”""","boat crew","boat Live N Hope","boat of Al Hattersly","""Boat of Captain Forman White,"boat of Scot's College rowers""","boat of Thomas Baker","boat of Wally Gibbons","Boat owned by Ricardo Laneiro","Boat with tourists onboard","boat x 2",'boat, occupants, P. Groenwald and others','boat, occupants: Jacob Kruger & crew','boat: Lady Charlotte, occupants: C. McSporran & his crew','Crew of Anti-submarine Squadron 23','crew','Haitian refugees perished when their boat capsized in choppy seas','males','males (wearing armor)','males, shark fishermen','mate & crew','multiple bathers','multiple boats including B.J. C. Brunt','No details','Occupants of skin boats','Passenger & crew','Passenger ferry Norman Atlantic','passenger in an automobile','pilot boat, occupants; Captain McAlister & crew','rowboat, ','rowboat, occupants: refugees fleeing Cuba','rowboats attacked by sharks','sailors','several Somali children','Severed human foot washed ashore (in sneaker with 2 Velcro closures)','ski-boat','skiboat','skiff, occupants: Russel J. Coles and others','slaves','surf patrol boat','Theodore Anderson’s captain & rest of crew taken by sharks']

zero_pers=['Jeff Wells claimed he rescued his "daughter" from a 4 m tiger shark']

one_pers=['After 2 days, Ann Dumas, 7,5 months pregnant, died of exposure & exhaution& her body was lashed to raft.','7.2 m boat. Occupant Kelvin Travers','5.75 m wooden boat, occupant: Yoshaiaki Ueda','5.4 m boat, occupant: Ivan Anderton','5 m skiboat; Stephanie','4.9 m fibreglass boat. Occupant: Jack Siverling',"""4.5 m boat, occupant: Rodney Lawn""","4.3 m skiff, occupant: Bob Shay","4-m runabout. Occupant: Allen Gade","""24' boat Shark Tagger Occupant Keith Poe""","22-ft boat.  Occupant Captain Scott Fitzgerald","""17' boat, occupant:  Richard Wade""","""16' skiff, occupant: W.A. Starck, II""","""16' Dreamcatcher. Occupant: Ian Bussus""","""16' boat, occupant: W. Lonergan""","""15-foot boat: occupant Woodrow Smith""","""14' catamaran, occupant: M. Leverenz""","""12' skiff, occupant: E.R.F. Johnson""","""12' boat Sio Ag, occupant: John Riding""",'12 m fishing boat. Occupant: Henry Tervo',"""10' rowboat, occupant: John Stephensen""",'British sailor from the  F-107','Arthur E. Taylor, a navy diver & member+G1053 of a 24-man demolition team',"""boat: 48' charter boat  GonFishin V""","""boat: 69.5' trawler Christy Nichole""",'British sailor from the  F-107','Robert W. McGhee, Private 1st Class, 8th Infantry','Sailor of tuna vessel No.12 Taiyo Marei','U.S. soldier in 161st Infantry Regiment, 25th Infantry Division','William Mills, a British solider, 36th Regiment']

two_pers=['a male & a female','a launch, occupants- Albert Cree & John Blacksall','8m inflatable boat. Occupants: Bhad Battle & Kevin Overmeyer','7 m boat, occupants: Tara Remington & Iain Rudkin','6 m boat: occupants  Stephen & Andrew Crust','5.4 m fibreglass boat, occupants: Robert & James Hogg','5 m skiboat Stephanie, occupants: Fanie Schoeman and Brigadier Bronkhorst','5 m skiboat Graanjan, occupants: Rudy van Graan, Jan de Waal Lombard','5 m boat, occupants: Don & Margaret Stubbs','5 m aluminum dinghy - occupants Mr. & Mrs. Paul Vickery',"""4.3 m skiff: occupants: James L. Randles, Jr. & James Myers""","""35' motor launch, occupants: Bill Fulham & T. Fanning""","""35' cruiser, Maluka II, occupants: Mr & Mrs E. Potts""","""3.5 -metre fibreglass boat, occupants: Harry Ulbrich and another fisherman""","""28' sea skiff, occupants: Alan Moree and another fisherman""","""22' boat, occupants: Saul White & Charles Dillione""","""20' boat; occupants: John Wright & a friend  ""","""19' clinker-built craft. Occupants: Ray Peckham & Mr. L.C. Wells""","""19' boat, occupants: Ray Peckham, L.C. Wells""","""18' launch, occupants: 2 fishermen""","""18' boat, occupants William & Leslie Newton""","""18' boat, occupants Richard Gunther & Donald Cavanaugh""","""18 hp Boston Whaler boat, occupant: G. W. Bane, Jr.""","""17' fishing launch, occupants: A. Burkitt & C. Brooke""","""17' fishing boat; occupants 2 men""","""16' motor launch owned by A. & E. Norton""","""14' open boat: occupants Richard Crew & Bob Thatcher""","""14' dinghy, 2 occupants""","""14' boat, occupants: 2 men""","""12' to 14' dory, occupants: John D. Burns & John MacLeod""","""12' ski, occupants: Bill Dyer & Cliff Burgess ""","""12' open motor boat, occupants Jack Platt & Peter Keyes""","""12' boat,           2 occupants""",'12-foot dinghy Occupants: R. Hunt & a friend.',"""11.6 m fibreglass boat. Occupants: Tony DeCriston & Dan Fink""","""10' skiff. Occupants F. Whitehead & L. Honeybone""","""10' row boat occupants;  Douglas Richards & George Irwin""",'boat, occupants:  Andrew Peterson & Peter Jergerson','boat, occupants:  Mike Taylor & his son, Jack, age 9','boat, occupants: Alf Dean, Jack Hood & Otto Bells','boat, occupants: Boyd Rutherford & Hamish Roper','boat, occupants: C. Nardelli & son','boat, occupants: John Griffiths & Thomas Johnson','boat, occupants: P.D. Neilly & Charlton Anderson','boat, Occupants: William Smith & Thomas Martin','boat; occupants: T & G Longhi','boat:  occupants: Nazzareno Zammit & Emmanuel',"""Boat: 14' Sunfish. Occupants Josh Long &  Troy Driscoll'""",'Boat: occupants: David Lock & his father','Boat: occupants: Tim Watson & Allan de Sylva','Bombardier J. Hall, Private Green of the Sherwood Foresters & Captain C. O. Jennings, R.E. Anti-tank Regiment','Brian Sierakowski & Barney Hanrahan','Bruce Flynn & his dive buddy','canoe, occupants: Chris Newman & Stewart Newman','canoe. Occupants: Doreen Tyrell & Frederick Bates','Captain Baxter & Dick Chares','Carlos Humberto Mendez & Esteban Robles','Colleen Chamberlin & Scott Chamberlin','Colonel B. & Sub-Lieutenant D.','Conway Plough &  Dr. Jonathan Higgs','crayfish boat. Occupants: Dave & Mitchell Duperouzel:','Curran See & Harry Lake','Dave Hamilton-Brown & Ant Rowan','dinghy, occupants: aborigine & lighthouse keeper','dinghy, occupants: Willem & Jan Groenwald','Dinghy. Occupants: Jeff Kurr and Andy Casagrande','Earl Yager & Riley McLachlan','Elena Hodgson & Isaac Ollis','Emil Uhlbrecht & unidentified person','Ensio Tiira & Fred Ericsson, deserters from the French Foreign Legion','Fishing boat. Occupants: Yunus Potur & Ali Durmaz','Gene Franken & Maurice McGregor','hobiecat, occupants: Judy Lambert  & a friend','Ida Parker & Kristen Orr','inflatable boat, occupants: Rudolf Bokelmann and Sakkie Vermeulen','inflatable dinghy, occupants: Ben Cropp, J. Harding & T. Fleischman','inflatable dinghy, occupants: Craig Ward & Gavin John Halse','"Inflatable kayak Occupants:  Andrej Kultan & Steve Hopkins."',"""inflatable rescue boat. Occupants: Lauren Johnson &. Kris O'Neill""",'J.T. Hales and Kenneth J. Hislop, Australian Navy frogmen','James C. Beason & Calvin E. Smith, Jr.  spent 16 hours in life raft','John Parker & Edward Matthews','Juan & Alex Bueno','Karl Pollerer & Eric Eisesenid','Lobster boat, occupants: Mr. P. Valentine & Mr. J. van Schalkwyk ','Louis Zamperini  & Russell Phillips','Mayabrit, an ocean rowing boat. Occupants: Andrew Barnett & J.C.S. Brendana','Monte Robinson & Andrew McNeill','Mr. Child & a Kanaka','Ned & Pawn','Nicaso Balanoba & Julian Dona','Occupants: Ivan Angjus & Stevo Kentera','occupants: John Chandler & Walter Winters','Occupants: Luke Jones & James Sequin','Occupants: Scott & John Fulton','Philip Case & William B. Gray','plywood dinghy, occupants: Jack Deegan & Trevor Millett','Pollione Perrini & Fioravante Perini','rowboat, occupants: Bob Scott & John Blackwell ','rowboat, occupants: James Mitchell-Hedges & Raymond McHenry','Salvatore & Agostino Bugeja','Sgt James Lacasse & Sgt David Milsten, USAF divers','skiboat, occupants: Danie & Fanie Schoeman','skiff with Dr. William T. Healey, Dr. Henry Callahan on board','skiff, occupants: J. & A. Ayerst','Teresea Britton (on raft) & a man (on floating debris)','Tony Moolman and another surfer','Two stowaways on German steamer Vela','Waade Madigan and Dr Seyong Kim','wooden boat, occupants: Jack Bullman & Keith Campbell']

three_pers=['Andong & 2 others','a skill. Occupants George Lunsford & 2 companions','6 m skiboat, occupants: Terry McManus, Dan Clark and Blackie Swart','6 m Seaduce - Occupants: Allen Roberts, Jason Savage & Rob Lindsay.','6 m boat Suki Saki, occupants: E.C. Landells & 2 friends','5.5 m fibreglass boat, occupants: Steven Piggott and Kelvin & Brendan Martin','5 m aluminum boat, occupants: Ben Turnbull, Lia & Neville Parker',"""4 m dinghy, occupants: Cecil Holmes, Chris Augustyn & Allen Varley ""","3.8-m boat with 3 people on board","""21' boat sank. Occupants: Max Butcher, George Hardy & Peter Thorne""","2.4 m rowboat, occupants: Edgar Brown, Jerry Welz & Cornelius  Stakenburg","""17' fishing boat. Occupants, Bubba DeMaurice, his wife & daughter""","""13' dinghy, occupant S. Smith, Leenee Dee & Marie Farmer   ""","""12' boat. Occupants:  Capt. E.J. Wines, Maj. W. Waller & Larry Waller""",'boat, occupants; Carl Sjoistrom & 2 other crew','boat, occupants:  Nels Jacobson & Franklin Harriman Covert','boat, occupants: Captains Charles Anderson, Emit Lindberg & Oscar Benson','Boat: occupants: Matt Mitchell & 2 other people','Burgess & 2 seamen','Captain Angus Brown, his son & brother','Captain Eric Hunt, the cook & a French passenger','catboat. Occupants: Captain Tuppe & 2 young ladiesr','Chris Haenga, Wayne Rangihuna  & Tamahau Tibble','D. R. Nelson, J. Greenberg & S.H. Gruber','Fishing boat  Bingo III , occupants: Michael Perkins, George Hornack & Capt. Lonergan ','"Fishing boat.   occupants: Laz Hüseyin, Ali Osman & Tursun "','Josh Francou, Michael Brister & Paul Bahr','launch, occupant Clive Whitrow, Dick Kuhne & Jim Pergola','Motor boat, occupants: Quintus Du Toit, J.H. van Heerden & J.P. Marais','Occupants: Andrew & Ben Donegan & Joel Ryan, ','Occupants: Jack Munro, Quinton Graham & Donald Shadler','open boat, occupants: Robert Ruark, Hoyle Dosher & Elmer Adkins','Rowgirls, an ocean rowing boat. Occupants: Sally Kettle, Claire Mills & Sue McMillan','Storm King; occupants - George Bridge & 2 sons','William Clinton Carter, Jr. & 2 other men']

four_pers=["""A 'tinnie". Occupants :Paul Sweeny, Paul Nieuwkerdk, John and Mark Kik """,'7 m skiboat Alrehmah III, occupants: Adolph Schlechter & 3 friends','6 m skiboat, occupants: Alex Mamacos, Noel Glintenkamp, Tony Mountifield & Dillon Alexandra','4.8 m skiboat, occupants: C. Cockroft & 3 men','16-foot launch, occupants: George Casey, Jack Byrnes, Julian Reynolds & Denny Laverty','Bry & David Mossman & 2 friends','Claude Hadley, William Grundy, Albert Faulkner & Frederick Faulkner','Fishing vessel. Occupants Gerry Malabago, Mark Anthony Malabago & 2 others','Inflatable dinghy, occupants: Jasmine Wigley, Greg Wilkie, Phil Rourke & Fleur Anderson','Mini Haa Haar, fiberglass boat, occupants: William Catton, Anthony Green, Tony & Kylie Barnes','rowboat, occupant: Joe Whitted, Christopher Quevedo & 2 Willard Brothers','skiboat, occupants: Gustav Boettger, Clive Mason, Keith Murrison & Sweis Olivier','William Olsen, William Peterson, Albert Thomas & R. Zekoski']

five_pers=['Albert Battles, James Dean & 4 crew','4.8-metre skiboat, Occupants: Rod Salm & 4 friends','boat of Dennis Kemp & 4 other occupants','boat Sea Hawk, occupants: R. Roberts & 4 others','boat, occupants:  Mr. Goslin & 4 passengers','Crew of aircraft: McGreevy, Beakley, Rosenthal, Ryan & Hodge','fishing boat, occupants: Simon Hlope & 4 other men','skiboat Double One, occupants: Anton & Michelle Gets, Ray Whitaker, John & Lyn Palmer','surf boat, occupants: Ray Sturdy and 4 other fishermen']

six_pers=['7 m fishing boat Metoo, occupants: Nicky & Paul Goles & 4 friends','6 m catamaran, occupants: Peter Robertson, Beauchamp Robertson, Gerald Spence & 3 crew','5 m inflatable boat, occupants: Kobus Potgieter & 5 friends',"""4.8 m  boat Peggy, occupants: John Oktober, L.A. van Zyl and 4 others""",'dinghy, occupants: T. Shipston, T. Whitta, L Cox, T. Jones, R. Genet & W. Pearce','motor boat, occupants: Mr. & Mrs. Sidney M. Colgate and their three children, Bayard, Caroline and Margaret ','Occupants: Hamza Humaid Al Sahra’a & 5 crew',"""Paul Timothy Lovette, Dr. Neal Beardsley, James C. Russell, Harold H. Mackie, Dale Howard & Diego J. Terres on 42' Navy craft, Marie"""]

seven_pers=['boat, occupants: Joseph Fitzback & 6 passengers','The June, occupants Bunny Pendelbury and crew of 6']

eight_pers=['Wes Wiggins and 7 others on the boat, Sparetime','7.5 m boat, occupants: 8 men','8 US airmen in the water, 1 was bitten by a shark']

for u in unknown:

    data['Number of victims']=data['Number of victims'].replace(u,'Unknown')

for one in one_pers:

    data['Number of victims']=data['Number of victims'].replace(one,1)

for two in two_pers:

    data['Number of victims']=data['Number of victims'].replace(two,2)

for three in three_pers:

    data['Number of victims']=data['Number of victims'].replace(three,3)

for four in four_pers:

    data['Number of victims']=data['Number of victims'].replace(four,4)

for five in five_pers:

    data['Number of victims']=data['Number of victims'].replace(five,5)

for six in six_pers:

    data['Number of victims']=data['Number of victims'].replace(six,6)

for seven in seven_pers:

    data['Number of victims']=data['Number of victims'].replace(seven,7)

for eight in eight_pers:

    data['Number of victims']=data['Number of victims'].replace(eight,8)

data['Number of victims']=data['Number of victims'].replace('boat Swift, occupants: Dolly Samuels & 8 other men',9)

data['Number of victims']=data['Number of victims'].replace('"21 m sportfishing vessel Seabiscuit. Occupants: Captain Louie Abbott & 2 dozen anglers"',25)

data['Number of victims']=data['Number of victims'].replace('"""15 Royal Canadian Airforce crew & 1 passenger"""',16)

data['Number of victims']=data['Number of victims'].replace('135 passengers & 13 crew',148)

data['Number of victims']=data['Number of victims'].replace('No survivors. 189 people were lost',295)

data['Number of victims']=data['Number of victims'].replace('Most were women & children',42)

data['Number of victims']=data['Number of victims'].replace('Mamerto Daanong, Tomas Inog & others',92)

data['Number of victims']=data['Number of victims'].replace('male + 20',21)

data['Number of victims']=data['Number of victims'].replace('At least 29 (and possibly another 71) Somali & Ethiopian refugees',90)

data['Number of victims']=data['Number of victims'].str.extract('(\d+)')
data['Number of victims'].fillna(1, inplace=True)

data['Number of victims']=data['Number of victims'].astype(int)

set(data['Number of victims'])
set(data['Name'].sample(n=40))
data.rename(columns={'Sex ': 'Sex'}, inplace=True)

data.columns = data.columns.str.strip()
def correct_name(row):

    if row["Name"] == "girl" or row["Name"] == "Girl":

        return "F"

    elif row["Name"] == "boy" or row["Name"] == "Boy":

        return "M"

    elif row["Name"] == "male" or row["Name"] == "Male":

        return "M"

    elif row["Name"] == "female" or row["Name"] == "Female":

        return "F"    

    else:

        return

    

data["name_corrected"] = data.apply(correct_name, axis=1)

data['Sex'].fillna(data.name_corrected, inplace=True)
def error_sex(row):

    if row['name_corrected']=='F':

        return 'F'

    elif row['name_corrected']=='M':

        return 'M'

    else:

        return row['Sex']

data['Sex']=data.apply(error_sex, axis=1)
set(data['Sex'])
data['Sex'] = data['Sex'].str.replace('N', 'M')

data['Sex'] = data['Sex'].str.strip()

data['Sex'] = data['Sex'].fillna('Unknown')

data['Sex'] = data['Sex'].str.replace('lli', 'Unknown')

data['Sex'] = data['Sex'].str.replace('.', 'Unknown')

set(data['Sex'])
data['Name'].fillna('Unknown', inplace=True)

data.loc[data['Name'].str.contains('male',na=False),'Name']='Unknown'

data.loc[data['Name'].str.contains('boy',na=False),'Name']='Unknown'

data.loc[data['Name'].str.contains('female',na=False),'Name']='Unknown'

data['Name']=data['Name'].str.replace('Anonymous','Unknown')

data['Name']=data['Name'].str.replace('Male','Unknown')

data['Name']=data['Name'].str.replace('fisherman','Unknown')

data['Name']=data['Name'].str.replace('Female','Unknown')

data['Name']=data['Name'].str.replace('crewman','Unknown')

data['Name']=data['Name'].str.replace('a young Scotsman','Unknown')

data['Name']=data['Name'].str.replace('a native ','')

data['Name'] = data['Name'].replace('\s+', ' ', regex=True)
set(data['Type'])
data['Type'] = data['Type'].str.strip()

data['Type'] = data['Type'].str.replace('Boating','Boat')

data['Type'] = data['Type'].str.replace('Invalid','Unknown')

print(set(data['Type']))
print(set(data['Country']))
data['Country'].fillna('Unknown', inplace=True)

data['Country'] = data['Country'].str.strip().str.upper()

print(set(data['Country']))
print(set(data['Species'].sample(n=20)))
data['Species'].fillna('Unknown', inplace=True)
data.drop(['Year_','name_corrected','test'], axis=1, inplace=True)
data.head()
data = data[['Case Number','Day','Month','Year','Date comment','Type','Country','Area','Location','Activity','Name','Sex','Age','Injury','Fatal (Y/N/U)','Number of victims','Species','Investigator or Source','href']]
data.to_csv('GSAF5_clean.csv', index=False)