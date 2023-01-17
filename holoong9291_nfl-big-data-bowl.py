import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KernelDensity



import warnings

warnings.filterwarnings("ignore")



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
# 普通的随机森林模型构建的分裂节点时是随机选取特征的，极端随机森林在构建每一棵树的分裂节点时，

# 不会任意的选取特征，而是先随机收集一部分特征，然后利用信息熵/基尼指数挑选最佳的节点特征；

# 这使得它可以用更少的树达到比随机森林更优的效果，但是过拟合的风险会大一些；

class RegressorConditional:

    def get_o_cat(self, o):

        return np.sum([o>pct for pct in self.percentiles], axis=0)

    # 默认使用极端随机森林、1000棵树、全部线程、启用bootstrap和obb_score，

    # obb是out-of-bag，意思是用集外数据验证模型性能，因此不需要交叉验证，这个bagging类算法的优势

    def __init__(self, model=ExtraTreesRegressor(n_estimators=500, n_jobs=-1, bootstrap=True, oob_score=True)):

        self.model = model

    def fit(self, X, y):

        targ = np.where(y>=0, np.log(1+np.abs(y)), -np.log(1+np.abs(y)))

        self.model.fit(X, targ)

        o = self.model.oob_prediction_

        self.percentiles = np.percentile(o, list(range(10, 100, 10)))

        o_cat = self.get_o_cat(o)

        self.dist = {}

        for oc in range(len(self.percentiles) + 1):

            filt = [oi==oc for oi in o_cat]

            kde = KernelDensity(kernel='exponential', metric='manhattan', bandwidth=0.3)

            kde.fit(list(zip(y[filt])))

            self.dist[oc] = np.exp(kde.score_samples(list(zip(range(-99, 100)))))

            self.dist[oc] /= sum(self.dist[oc])

    def predict_proba(self, X):

        o = self.model.predict(X)

        o_cat = self.get_o_cat(o)

        return np.array([self.dist[oc] for oc in o_cat])
df_train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', parse_dates=['TimeHandoff','TimeSnap'], infer_datetime_format=True, low_memory=False)
df_train.info()
df_train.isnull().sum()/len(df_train)
df_train[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']] = df_train[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']].fillna(method='ffill')

df_train[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']].isnull().sum()
df_train[pd.isnull(df_train.StadiumType)].Stadium.unique()
df_train.groupby('StadiumType').size().sort_values()
# fill with outdoor

df_train[df_train.Stadium=='StubHub Center'].StadiumType.unique()
# fill with outfoor

df_train[df_train.Stadium=='MetLife Stadium'].StadiumType.unique()
# fill with ffill

df_train[df_train.Stadium=='TIAA Bank Field'].StadiumType.unique()
def fill_stadiumtype(row):

    if row['Stadium'] in ['StubHub Center','MetLife Stadium'] and pd.isnull(row['StadiumType']):

        return 'Outdoor'

    return row['StadiumType']



df_train.StadiumType = df_train.apply(fill_stadiumtype, axis=1)

df_train.StadiumType = df_train.StadiumType.fillna(method='ffill')
df_train[pd.isnull(df_train.StadiumType)].Stadium.unique()
df_train[pd.isnull(df_train.FieldPosition)][['X','Y','PlayDirection','PossessionTeam','Team','HomeTeamAbbr','VisitorTeamAbbr','FieldPosition']].sample(10)
df_train.FieldPosition = df_train.FieldPosition.fillna('Middle')
df_train.groupby('OffenseFormation').size().sort_values()
df_train.OffenseFormation = df_train.OffenseFormation.fillna('SINGLEBACK')
df_train[df_train.Team=='home'][df_train.HomeTeamAbbr=='BAL'][df_train.VisitorTeamAbbr=='NO'][['Team','HomeTeamAbbr','VisitorTeamAbbr','DefensePersonnel','DefendersInTheBox']].sample(10)
defendersInTheBox = df_train.groupby(['Team','HomeTeamAbbr','VisitorTeamAbbr','DefensePersonnel']).DefendersInTheBox.median()

defendersInTheBox
def fill_defendersinthebox(row):

    if pd.isnull(row['DefendersInTheBox']):

        return defendersInTheBox[row['Team']][row['HomeTeamAbbr']][row['VisitorTeamAbbr']][row['DefensePersonnel']]

    return row['DefendersInTheBox']



df_train.DefendersInTheBox = df_train.apply(fill_defendersinthebox, axis=1)
df_train.Orientation = df_train.Orientation.fillna(df_train.Orientation.mean())

df_train.Dir = df_train.Dir.fillna(df_train.Dir.mean())
df_train.isnull().sum()
object_columns = {}

for col in df_train.columns:

    if df_train[col].dtype=='object':

        object_columns[col] = df_train[col].value_counts().index[0]
stadiumtype_map = {

    'Outdoor':'Outdoor','Outdoors':'Outdoor','Outddors':'Outdoor','Oudoor':'Outdoor','Ourdoor':'Outdoor','Outdor':'Outdoor','Outside':'Outdoor',

    'Indoors':'Indoor','Indoor':'Indoor',

    'Retractable Roof':'Retractable Roof',

    'Retr. Roof-Closed':'Retr. Roof-Closed','Retr. Roof - Closed':'Retr. Roof-Closed','Retr. Roof Closed':'Retr. Roof-Closed',

    'Retr. Roof-Open':'Retr. Roof-Open','Retr. Roof - Open':'Retr. Roof-Open',

    'Open':'Open',

    'Indoor, Open Roof':'Indoor, Open Roof',

    'Indoor, Roof Closed':'Indoor, Roof Closed',

    'Outdoor Retr Roof-Open':'Outdoor Retr Roof-Open',

    'Dome':'Dome','Domed':'Dome',

    'Domed, closed':'Domed, closed','Closed Dome':'Domed, closed','Dome, closed':'Domed, closed',

    'Domed, Open':'Domed, Open','Domed, open':'Domed, Open',

    'Heinz Field':'Heinz Field',

    'Cloudy':'Cloudy',

    'Bowl':'Bowl',

}
df_train.StadiumType = df_train.StadiumType.map(stadiumtype_map)
possessionteam_map = {

    'BLT':'BAL',

    'CLV':'CLE',

    'ARZ':'ARI',

    'HST':'HOU'

}

df_train.PossessionTeam = df_train.PossessionTeam.apply(lambda pt:possessionteam_map[pt] if pt in possessionteam_map.keys() else pt)
turf_map = {

    'FieldTurf':'Field Turf','Field turf':'Field Turf',

    'UBU Speed Series-S5-M':'UBU-Speed Series-S5-M',

    'Twenty-Four/Seven Turf':'Twenty Four/Seven Turf',

    'natural grass':'Natural grass',

    'Field turf':'Field Turf',

    'Natural Grass':'Natural grass','Naturall Grass':'Natural grass',

    'grass':'Grass',

    'Artificial':'Artifical',

    'FieldTurf360':'FieldTurf 360'

}

df_train.Turf = df_train.Turf.apply(lambda pt:turf_map[pt] if pt in turf_map.keys() else pt)
location_map = {

    'Foxborough, MA':'Foxborough',

    'Orchard Park NY':'Orchard Park','Orchard Park, NY':'Orchard Park',

    'Chicago. IL':'Chicago','Chicago, IL':'Chicago',

    'Cincinnati, Ohio':'Cincinnati','Cincinnati, OH':'Cincinnati',

    'Cleveland, Ohio':'Cleveland','Cleveland, OH':'Cleveland','Cleveland,Ohio':'Cleveland','Cleveland Ohio':'Cleveland','Cleveland':'Cleveland','Cleveland Ohio':'Cleveland',

    'Detroit, MI':'Detroit','Detroit':'Detroit',

    'Houston, Texas':'Houston','Houston, TX':'Houston',

    'Nashville, TN':'Nashville',

    'Landover, MD':'Landover',

    'Los Angeles, Calif.':'Los Angeles','Los Angeles, CA':'Los Angeles',

    'Green Bay, WI':'Green Bay',

    'Santa Clara, CA':'Santa Clara','Santa Clara, CSA':'Santa Clara',

    'Arlington, Texas':'Arlington','Arlington, TX':'Arlington',

    'Minneapolis, MN':'Minneapolis',

    'Denver, CO':'Denver','Denver CO':'Denver',

    'Baltimore, Md.':'Baltimore','Baltimore, Maryland':'Baltimore',

    'Charlotte, North Carolina':'Charlotte','Charlotte, NC':'Charlotte','Charlotte North Carolina':'Charlotte',

    'Indianapolis, Ind.':'Indianapolis',

    'Jacksonville, FL':'Jacksonville','Jacksonville, Fl':'Jacksonville','Jacksonville, Florida':'Jacksonville','Jacksonville Florida':'Jacksonville',

    'Kansas City, MO':'Kansas City','Kansas City,  MO':'Kansas City',

    'New Orleans, LA':'New Orleans','New Orleans, La.':'New Orleans','New Orleans':'New Orleans',

    'Pittsburgh':'Pittsburgh','Pittsburgh, PA':'Pittsburgh',

    'Tampa, FL':'Tampa',

    'Carson, CA':'Carson',

    'Oakland, CA':'Oakland',

    'Seattle, WA':'Seattle','Seattle':'Seattle',

    'Atlanta, GA':'Atlanta',

    'East Rutherford, NJ':'East Rutherford','E. Rutherford, NJ':'East Rutherford','East Rutherford, N.J.':'East Rutherford',

    'London, England':'London','London':'London',

    'Philadelphia, Pa.':'Philadelphia','Philadelphia, PA':'Philadelphia',

    'Glendale, AZ':'Glendale',

    'Foxborough, Ma':'Foxborough',

    'Miami Gardens, Fla.':'Miami Gardens','Miami Gardens, FLA':'Miami Gardens','Miami Gardens, FL':'Miami Gardens',

    'Mexico City':'Mexico City',

    

}

df_train.Location = df_train.Location.apply(lambda pt:location_map[pt] if pt in location_map.keys() else pt)
df_train['TeamBelongAbbr'] = df_train.apply(lambda row:row['HomeTeamAbbr'] if row['Team']=='home' else row['VisitorTeamAbbr'],axis=1)
df_train['Offense'] = df_train.apply(lambda row:row['PossessionTeam']==row['TeamBelongAbbr'],axis=1)
plt.figure(figsize=(30, 50))

subplot_len = len(df_train[df_train.GameId==2017090700].groupby(['GameId','PlayId']))

df_train_groupby_gp = df_train[df_train.GameId==2017090700].groupby(['GameId','PlayId'])

i=1



for gp,chance in df_train_groupby_gp:

    game_id,play_id = gp[0],gp[1]

    rusher = chance[chance.NflId==chance.NflIdRusher]

    offense = chance[chance.Offense]

    defense = chance[~chance.Offense]

    yard_line_left = offense.YardLine.iloc[0]+10 # yard_line 加10偏移量，这个10是左侧的达阵区

    yard_line_right = offense.YardLine.iloc[0]+2*(50-offense.YardLine.iloc[0])+10

    yard_line = yard_line_left if np.abs(yard_line_left-rusher.X.iloc[0])<=(yard_line_right-rusher.X.iloc[0]) else yard_line_right

    

    plt.subplot(subplot_len/4 if (subplot_len/4*4)==subplot_len else (subplot_len/4)+1,4,i)#, sharex=True, sharey=True)

    plt.xlim(0,120)# 0~120已经包含了达阵区，实际场内只有100码，码线也是0~100的范围

    plt.ylim(-10,63)

    plt.scatter(list(offense.X),list(offense.Y),marker='x',c='red',s=20,alpha=0.5,label='Offense-'+offense.Team.iloc[0]+'-'+offense.TeamBelongAbbr.iloc[0])

    plt.scatter(list(defense.X),list(defense.Y),marker='o',s=18,alpha=0.5,label='Defense-'+defense.Team.iloc[0]+'-'+defense.TeamBelongAbbr.iloc[0])

    plt.scatter(list(rusher.X),list(rusher.Y),marker='<' if offense.PlayDirection.iloc[0]=='left' else '>',c='black',s=50,label='Rusher')

    plt.plot([yard_line,yard_line],[-100,100],c='orange')

    

    plt.plot([10,10],[-100,100],c='green',linewidth=3) # down zone left

    plt.plot([110,110],[-100,100],c='green',linewidth=3) # down zone right

    plt.title('Quarter:'+str(offense.Quarter.iloc[0])+' - '+str(offense.GameClock.iloc[0])+' - '+offense.PlayDirection.iloc[0]+' - push:'+str(offense.Yards.iloc[0])+',dis:'+str(offense.Dis.iloc[0])+',need:'+str(offense.Distance.iloc[0])+' DF:'+str(defense.DefendersInTheBox.iloc[0]))

    plt.legend()

    

    i+=1



plt.show()
df_train.info()
#df_train = df_train.drop(['DisplayName','JerseyNumber','WindSpeed','WindDirection'], axis=1)

df_train = df_train.drop(['WindSpeed','WindDirection'], axis=1)

#df_train.PossessionTeam = df_train.apply(lambda row:1 if row['PossessionTeam']==row['TeamBelongAbbr'] else 0, axis=1)

#df_train.FieldPosition = df_train.apply(lambda row:1 if row['FieldPosition']==row['TeamBelongAbbr'] else 0, axis=1)

df_train.DefendersInTheBox = df_train.DefendersInTheBox.astype('int8')

df_train.PlayerHeight = df_train.PlayerHeight.apply(lambda height:int(height[0])*12+int(height[2:])).astype('int')

df_train['Age'] = df_train.PlayerBirthDate.apply(lambda bd:2019-int(bd[-4:]))

df_train = df_train.drop(['PlayerBirthDate'], axis=1)
df_train['TimeFromSnapToHandoff'] = (df_train.TimeHandoff - df_train.TimeSnap).apply(lambda x:x.total_seconds()).astype('int8')
df_train['GameDuration'] = (df_train.GameClock.apply(lambda gc:15*60-int(gc[:2])*60-int(gc[3:5]))) + (df_train.Quarter-1)*15*60
def split_pos(poss,POS):

    count = 0

    for pos in [poss['OffensePersonnel'],poss['DefensePersonnel']]:

        for p in pos.split(','):

            p = p.strip()

            space_idx = p.find(' ')

            count_ = p[:space_idx]

            pos_ = p[space_idx+1:]

            if pos_==POS:

                count+=int(count_)

    return count



# POSITIONS 过滤了一部分未出现在训练数据中的

# POSITIONS = ['SS', 'DE', 'ILB', 'FS', 'CB', 'DT', 'WR', 'TE', 'T', 'QB', 'RB', 'G', 'C', 'OLB', 'NT', 'FB', 'MLB', 'LB', 'OT', 'OG', 'HB', 'DB', 'S', 'DL', 'SAF']

POSITIONS = ['WR', 'TE', 'QB', 'RB', 'LB', 'DB', 'DL']

for POS in POSITIONS:

    df_train['Position_'+POS] = df_train[['OffensePersonnel','DefensePersonnel']].apply(split_pos,args=(POS,),axis=1)





position_features = [col for col in df_train.columns if col.startswith('Position_')]



for col in position_features:

    if df_train[col].mean()<=0:

        df_train = df_train.drop(col, axis=1)

        del POSITIONS[POSITIONS.find(col[col.find('_')+1:])]

df_train = df_train.drop(['OffensePersonnel','DefensePersonnel'], axis=1)

        

position_features = ['Position_'+pos for pos in POSITIONS]



print(position_features)

print(df_train[position_features].sample(30))
# goal区：也就是码线对方半场10码或10码内，此时就处于goal区；

# YardLine: 1~50

# 对方半场、YardLine<=10，即为goal区

df_train['GoalZone'] = df_train[['FieldPosition','TeamBelongAbbr','YardLine']].apply(lambda pty:1 if pty['FieldPosition']!=pty['TeamBelongAbbr'] and pty['YardLine']<=10 else 0, axis=1)
# 首攻危险：down为4，且distance大于5；

df_train['FirstDownDanger'] = df_train[['Distance','Down']].apply(lambda dd:1 if dd['Down']>3 and dd['Distance']>5 else 0, axis=1)
# 距离达阵还有多少码

# 球场内总长为100码，通过码线、PossessionTeam、FieldPosition即可判断距离达阵的码数

df_train['DistanceTouchDown'] = df_train[['YardLine','FieldPosition','PossessionTeam']].apply(lambda yfp:100-yfp['YardLine'] if(yfp['PossessionTeam']==yfp['FieldPosition']) else yfp['YardLine'], axis=1)
# df_train['TeamBelongAbbr'] = df_train.apply(lambda row:row['HomeTeamAbbr'] if row['Team']=='home' else row['VisitorTeamAbbr'],axis=1)

# df_train['Offense'] = df_train.apply(lambda row:row['PossessionTeam']==row['TeamBelongAbbr'],axis=1)
DisplayNameLabels = {'Jay Bromley', 'Takkarist McKinley', 'Preston Smith', 'Thomas Rawls', 'Mark Ingram', 'Adam Shaheen', 'Tarik Cohen', 'Corey Clement', 'Kevin Minter', 'Byron Cowart', 'Shareece Wright', 'Terrell Edmunds', 'Tony Pollard', 'Duke Johnson', 'Greg Little', 'Fitzgerald Toussaint', "De'Angelo Henderson", 'Marvell Tell', 'Deon Bush', 'Aaron Colvin', 'Mike Wallace', 'Calais Campbell', 'Don Barclay', 'Elijhaa Penny', 'Kerry Wynn', 'Cody Davis', 'Todd Davis', 'Jake Ryan', 'Josh LeRibeus', 'T.J. Yates', 'Jeremiah Attaochu', 'Vincent Valentine', 'Keelan Doss', 'Donald Stephenson', 'Bryce Harris', "James O'Shaughnessy", 'Keion Crossen', 'Tommylee Lewis', 'Artie Burns', 'Joe Noteboom', 'Pharoh Cooper', 'Jon Hilliman', 'Luke Kuechly', 'Jimmy Moreland', 'Justin Reid', 'Joe Schobert', 'Lamarr Houston', 'Michael Wilhoite', 'Jeremiah Ledbetter', 'Rashaad Coward', 'Carson Palmer', 'Frank Clark', 'Marcus Burley', 'Corey Thompson', 'Akeem Hunt', 'Zach Brown', 'Byron Bell', 'Tommy Sweeney', "Cre'von LeBlanc", 'Mitch Unrein', 'Randy Gregory', "Adoree' Jackson", 'Josh Sitton', 'Daniel Brunskill', 'Brian Parker', 'Trey Walker', 'Troy Fumagalli', 'Will Parks', 'Clinton McDonald', 'Jordan Thomas', 'Morris Claiborne', 'Bradley Roby', 'Patrick Chung', 'Trey Flowers', 'Vance McDonald', 'KeiVarae Russell', 'Gabe Holmes', 'Michael Crabtree', 'Greg Mancz', 'Darryl Tapp', 'Troy Hill', 'AJ McCarron', 'Connor Williams', 'Hayes Pullard', 'Tim Patrick', 'Lamarcus Joyner', 'Bradley Chubb', 'Jalen Reeves-Maybin', 'Corey Ballentine', 'Joel Bitonio', 'Dalton Schultz', 'Ed Oliver', 'Trey Hopkins', 'Daryl Williams', 'Cordrea Tankersley', 'Jomal Wiltz', 'Vincent Rey', "Dont'a Hightower", 'Kapri Bibbs', 'Ryan Jensen', 'Justin Bethel', 'Kelvin Sheppard', 'Marlon Humphrey', 'Isaiah Ford', 'Marquise Brown', "De'Vante Harris", 'Darius Leonard', 'John Atkins', 'Eric Ebron', 'Landry Jones', 'Dylan Donahue', 'Logan Thomas', 'Tre McBride', 'Cooper Kupp', 'Michael Mauti', 'Doug Middleton', 'Marcell Harris', 'Jerry Tillery', 'Khari Lee', 'Robert Turbin', 'Robert Golden', 'Alex Cappa', 'Ufomba Kamalu', 'Jerome Cunningham', 'Garrett Sickels', 'A.J. Green', 'Sammy Watkins', 'Will Hernandez', 'Alfred Morris', 'Blake Bell', 'Elandon Roberts', 'Charles Sims', 'Michael Thomas', 'D.J. Jones', 'Jamal Adams', 'Jermaine Gresham', 'Isaiah Prince', 'Anthony Barr', 'Janoris Jenkins', 'Josh Sweat', 'Deonte Harris', 'Markus Golden', 'Olsen Pierre', 'Derrick Nnadi', 'Joshua Holsey', 'Brian Hill', 'Alex Erickson', 'Justin Coleman', 'Kyle Emanuel', 'Danny Isidora', 'Dre Kirkpatrick', 'Troy Niklas', 'Allen Hurns', 'Adam Butler', 'C.J. Beathard', 'Chris Hairston', 'Domata Peko', 'Barry Church', 'David Njoku', 'C.J. Prosise', 'Lorenzo Alexander', 'Jake Butt', 'Brian Burns', 'Anthony Zettel', 'Ted Larsen', 'Craig James', 'Sean Mannion', 'Deshazor Everett', 'Cameron Batson', 'Rico Gathers', 'Vic Beasley', 'Travaris Cadet', 'Alvin Kamara', 'Anthony Chickillo', 'Andrew Donnal', 'Jordan Whitehead', 'Davontae Harris', 'Brandon Williams', 'Justin Jackson', 'Chris Conte', 'Zaire Anderson', 'Jayron Kearse', 'LaTroy Lewis', 'Roy Robertson-Harris', 'Vince Williams', 'Kris Boyd', 'Stacy Coley', 'Weston Richburg', 'Isaiah McKenzie', 'Nate Orchard', 'Kayvon Webster', 'Jonathan Harris', 'Marcus Smith', 'Wes Schweitzer', 'Joe Haeg', 'Leonte Carroo', 'Shaq Calhoun', 'Kent Perkins', 'Nicholas Grigsby', 'Joe Vellano', 'Mike Daniels', 'Christian Kirk', 'Justin Pugh', 'Mike Mitchell', 'Mike Person', 'Isaac Seumalo', 'Jaire Alexander', 'Tracy Walker', 'Darius Powe', 'Paul Richardson', 'Auden Tate', 'Ola Adeniyi', 'Erik Harris', 'Michael Pierce', "A'Shawn Robinson", 'C.J. Moore', 'Jordan Evans', 'Mackensie Alexander', 'Albert McClellan', 'Andre Ellington', 'Duke Dawson', 'James Washington', 'Adam Humphries', 'Harry Douglas', 'Ronald Leary', 'Cameron Wake', 'Kevon Seymour', 'Chris Lacy', 'Leonard Johnson', 'Josh McCown', 'Jaleel Johnson', 'Shawn Williams', 'Will Compton', 'Jack Mewhort', 'Harrison Phillips', 'Tyvon Branch', 'Pasoni Tasini', 'B.J. Goodson', 'Marcus Cannon', 'Damarious Randall', 'Kevin Strong', 'Khalil Mack', 'Saeed Blacknall', 'Marshon Lattimore', 'Maurkice Pouncey', 'Quenton Nelson', 'Shelby Harris', 'Antony Auclair', 'Austin Hooper', 'Damien Wilson', 'LaRoy Reynolds', 'Taylor Heinicke', 'Terrance West', 'Deyshawn Bond', 'Chris Prosinski', 'Senorise Perry', 'Harrison Smith', 'Jacquizz Rodgers', 'Dontari Poe', 'Josh Kline', 'Travis Kelce', 'Tani Tupou', 'Chris Godwin', "Su'a Cravens", 'Brandon Dillon', 'Jordan Lucas', 'Earl Thomas', 'Josh Hill', 'Mike Boone', 'Trevor Davis', 'Chukwuma Okorafor', 'Joey Hunt', 'Chase Edmonds', 'Mohamed Sanu', 'Kevin Johnson', 'Kenneth Dixon', 'Rob Havenstein', 'Matt Paradis', 'Shamar Stephen', 'Jason Witten', 'Hunter Sharp', 'Josh Bynes', 'Josh Harvey-Clemons', 'Justin Hardee', 'Kevin White', 'Eric Berry', 'Colt McCoy', 'RJ McIntosh', 'Brandon Coleman', 'D.J. Humphries', 'Louis Murphy', 'Malik Jefferson', 'Steven Terrell', 'Shy Tuttle', 'Rees Odhiambo', 'Jonathan Bullard', 'Max Garcia', 'Greg Ward', "Ja'Whaun Bentley", 'Tony Brown', 'Julius Peppers', 'Eddie Goldman', 'Jay Ajayi', 'Elijah McGuire', 'Jamon Brown', 'Anthony Fasano', 'Colin Jones', 'Stephon Tuitt', 'Joe Dahl', 'Nolan Carroll', 'Mike Evans', 'Kentrell Brothers', 'Mitchell Schwartz', 'Durham Smythe', 'Vontae Davis', 'Darian Stewart', 'Taven Bryan', 'Austin Reiter', 'Rashan Gary', 'Jason Peters', 'Morgan Moses', 'Jelani Jenkins', 'Oren Burks', 'Ndamukong Suh', 'Matthias Farley', 'Brandon Mebane', 'James Burgess', 'Russell Shepard', 'Sealver Siliga', 'Lamar Miller', 'LeShun Daniels', 'Lawrence Guy', 'Ezekiel Turner', 'Steven Parker', 'Martrell Spaight', 'Demetrius Harris', 'Vernon Davis', 'Peyton Barber', 'Blake Countess', 'Malcolm Brown', 'Jordan Howard', 'Jake Brendel', 'Tim Lelito', 'Jonathan Stewart', 'Miles Sanders', 'Torry McTyer', 'Kareem Martin', 'Devin Bush', 'Caraun Reid', 'Greg Gaines', 'Karlos Dansby', 'Billy Price', 'Channing Ward', 'Jeremy Vujnovich', 'Alshon Jeffery', 'Trey Hendrickson', 'Quenton Meeks', 'Nigel Harris', 'Geremy Davis', 'Charone Peake', 'Cam Phillips', 'Aaron Rodgers', 'Josh Walker', 'Cooper Rush', 'Ricky Jean Francois', 'Ronald Blair', 'Emanuel Byrd', 'Aaron Neary', 'Allen Barbre', 'Corey Coleman', 'Charcandrick West', 'John Miller', 'Chad Williams', 'Bryan Mone', 'Justin Durant', 'Josh Mauro', 'Matt Skura', 'Datone Jones', 'Nelson Agholor', 'Jack Crawford', 'Travin Howard', 'Carl Davis', 'Ryan Smith', 'Riley Bullough', 'Shaquill Griffin', 'Josh Forrest', 'Jeff Heath', 'Russell Bodine', 'Mack Brown', 'Elgton Jenkins', 'Ryan Murphy', 'Eric Lee', 'Colby Gossett', 'Anthony Miller', 'Reggie Gilbert', 'DaQuan Jones', 'Patrick Robinson', 'Malcolm Butler', 'Dwayne Allen', 'James Bradberry', 'Chris Baker', 'J.J. Arcega-Whiteside', 'Eddie Vanderdoes', 'Khalif Barnes', 'Laurent Duvernay-Tardif', 'Michael Roberts', 'Mike Jordan', 'Aaron Ripkowski', 'Rod Smith', 'Luke Joeckel', 'Jawaan Taylor', 'Andre Dillard', 'Kenny Clark', 'Deandre Coleman', 'Lavar Edwards', 'Ahmad Brooks', 'Adrian Amos', 'Abdullah Anderson', 'Taysom Hill', 'Aaron Donald', 'C.J. Uzomah', 'Shakial Taylor', 'J.J. Watt', 'Jesse Davis', 'Brandon Dixon', 'Russell Gage', 'Spencer Long', 'B.J. Finney', 'Jamal Agnew', 'Brian Hoyer', 'Matt Slauson', 'Matt Feiler', 'Bobby Okereke', 'Rashawn Scott', 'DeShawn Shead', 'Thomas Davis', 'Denzel Rice', 'Blair Brown', 'Andy Isabella', 'John Simon', 'Angelo Blackson', 'Deontay Burnett', 'George Odum', 'Justin Evans', 'Bobby McCain', 'Cam Newton', 'Matt Tobin', 'Jake Rudock', 'Bryan Braman', 'Trent Scott', 'Shamarko Thomas', 'Rasul Douglas', 'Brandon LaFell', 'Ricardo Louis', 'David Williams', 'Duron Harmon', 'Vontarrius Dora', 'Nick Martin', 'Curtis Grant', 'Josh Martin', 'Vadal Alexander', 'Eric Wilson', 'Chris Reed', 'Kamar Aiken', 'Duke Ejiofor', 'Tyler Eifert', 'Jalen Thompson', 'Brandon Facyson', 'Sam Shields', 'Jamarco Jones', 'Brian Schwenke', 'Linval Joseph', 'David Bass', 'Nick Mullens', 'Rasheem Green', 'Shalom Luani', 'Bryce Callahan', 'Malik Hooker', 'Bradley Bozeman', 'Cyrus Jones', 'Chris Hogan', 'Trumaine Johnson', 'Robert Nelson', 'Garrett Gilbert', 'Sean McGrath', 'Terrelle Pryor', 'Bradley Sowell', 'Ryan Tannehill', 'Detrez Newsome', 'Germaine Pratt', 'William Jackson', 'Jon Halapio', 'A.J. Francis', 'Ahmad Thomas', 'Emmett Cleary', 'Bronson Hill', 'Ameer Abdullah', 'Sam Jones', 'Roquan Smith', 'Emmanuel Ogbah', 'Pierre Garcon', 'Aqib Talib', 'Tyler Catalina', 'Kelvin Beachum', 'Jeremy Sprinkle', 'Johnathan Joseph', 'Tyler Higbee', 'Chris Herndon', 'Kyle Peko', 'Will Tye', 'Matt Dickerson', 'Theo Riddick', 'Anthony Lanier', 'Ukeme Eligwe', 'David DeCastro', 'Brian Mihalik', 'John Franklin-Myers', 'Justin Murray', 'Chase Allen', 'Luke Stocker', 'Zach Kerr', 'Chris Milton', 'Derek Carrier', 'Courtland Sutton', 'Ito Smith', 'Mario Edwards', 'Travis Rudolph', 'Cullen Gillaspia', 'Jalen Richard', 'Cornelius Washington', 'Carlos Thompson', 'Zach Sieler', 'Nick Vannett', 'Shaq Lawson', 'Mose Frazier', 'Charles Clay', 'Anthony Castonzo', 'Jason McCourty', 'Quincy Adeboyejo', 'Tyrann Mathieu', 'Buster Skrine', 'Danielle Hunter', 'Mike Iupati', 'Jordan Jenkins', 'Michael Liedtke', 'Abry Jones', 'Luke Bowanko', 'Jonathan Ledbetter', 'Jason Spriggs', 'Adarius Taylor', "Da'Norris Searcy", 'Jarrad Davis', 'Austin Davis', 'Patrick DiMarco', 'Kendall Sheffield', 'Uchenna Nwosu', 'Nate Stupar', 'Mason Cole', 'Sean Culkin', 'Dexter McDougle', 'Shane Smith', 'Greg Olsen', 'D.J. Foster', 'Brandon Dunn', 'Matt Kalil', 'T.J. Edwards', 'Leonard Fournette', 'Andrew Whitworth', 'Tyrique Jarrett', 'George Fant', 'Jabrill Peppers', 'Ethan Pocic', 'Danny Trevathan', 'Michael Deiter', 'Trenton Cannon', 'Ryan Malleck', 'Quinton Spain', 'Odell Beckham', 'Nick Williams', 'Chris Banjo', 'Will Dissly', 'Antoine Bethea', 'Quan Bray', 'Vernon Butler', 'Dante Fowler', 'Temarrick Hemingway', 'Charles Washington', 'Trey Pipkins', 'Levine Toilolo', 'D.J. White', 'Akeem Ayers', 'Ronnie Harrison', 'Ryan Schraeder', 'Josh Jackson', 'Mike Ford', 'James Hurst', 'Dee Ford', 'Kai Nacua', 'Cliff Avril', 'Sean Murphy-Bunting', 'Bobo Wilson', 'Jamiyus Pittman', 'Mason Schreck', 'Bruce Carter', 'Derrick Johnson', 'Romeo Okwara', 'John Cominsky', 'Brandon Banks', 'Domata Peko Sr.', 'Jason Vander Laan', 'DeAngelo Hall', 'Hayden Hurst', 'Freddie Martino', 'Ryan Lewis', 'Alexander Mattison', 'Corey Graham', 'Marshal Yanda', 'Isaiah Johnson', 'Joe Jackson', 'Trent Murphy', 'Obi Melifonwu', 'Ejuan Price', 'Vonn Bell', 'Vita Vea', 'Brandon Zylstra', 'Amari Cooper', 'Desmond Harrison', 'L.J. Collier', 'Dan McCullers', 'Tre Madden', 'Avery Williamson', 'Breshad Perriman', 'Jordan Taylor', 'Leger Douzable', 'Aviante Collins', 'Ross Cockrell', 'Clelin Ferrell', "J'Marcus Webb", 'Hassan Ridgeway', 'Junior Galette', 'Ziggy Hood', 'Lonnie Johnson', 'Zach Allen', 'John Hughes', 'DeShon Elliott', 'Larry Ogunjobi', 'Juston Burris', 'Treston Decoud', 'Larry Pinkard', 'Jermaine Kearse', 'T.J. Lang', 'Deante Burton', 'Phillip Lindsay', 'Dion Sims', 'Devin Funchess', 'David Morgan', 'Rod Streater', 'Richie James', 'Mike Edwards', 'Robert Woods', 'Martellus Bennett', 'Josh Tupou', 'Benson Mayowa', 'Jabaal Sheard', 'David Johnson', 'Eli Ankou', 'Duke Riley', 'Isaiah Crowell', 'James Develin', 'Cameron Erving', 'Raekwon McMillan', 'Dontavius Russell', 'Ryan Groy', 'Trey Quinn', 'T.J. Logan', 'Geron Christian', 'Derrius Guice', 'Jacob Hollister', 'Kendell Beckwith', 'Brent Celek', 'Fletcher Cox', 'Deshaun Watson', 'Trey Burton', 'Pernell McPhee', 'Bernard Reedy', 'Michael Jordan', 'Derek Anderson', 'Keith Kirkwood', 'Rashaan Melvin', 'Reuben Foster', 'Austin Traylor', 'Latavius Murray', 'Joejuan Williams', 'Allen Lazard', 'Clint Boling', 'Kyle Lauletta', 'JuJu Smith-Schuster', 'Anthony Fabiano', 'Arden Key', 'Danny Amendola', 'Garett Bolles', 'Jordy Nelson', 'Deion Jones', 'Giovani Bernard', 'Marcell Dareus', 'Baker Mayfield', 'Jordan Simmons', 'Devante Bond', 'Cameron Meredith', 'Corn Elder', 'Byron Jones', 'A.J. Cann', 'Tahir Whitehead', 'Adam Thielen', 'John Phillips', 'Christian Miller', 'Jaylon Smith', 'Sherrick McManis', 'Olivier Vernon', 'Ed Dickson', 'Marcedes Lewis', 'Jehu Chesson', 'Malachi Dupre', 'Sterling Shepard', 'Ben Ijalana', 'Daniel Ross', 'David Moore', 'James Harrison', 'Brice McCain', 'Andrew Sendejo', 'Von Miller', 'Frank Ragnow', 'Anthony Harris', "De'Vondre Campbell", 'Tra Carson', 'DeVante Bausby', 'Chase Daniel', 'Darqueze Dennard', 'Jeremy Langford', 'Ryan Shazier', 'Dexter McDonald', 'Gabe Jackson', 'Jordan Mills', 'Ron Parker', 'Everson Griffen', 'Brandon Shell', 'Avery Moss', 'Joel Heath', 'Sharif Finch', 'Kyle Allen', 'Sam Darnold', 'Brian Price', 'Darvin Kidsy', 'Jeremiah George', 'Cassanova McKinzy', 'Derwin James', 'Neal Sterling', 'Pat Sims', 'Billy Turner', 'Chris Jones', 'Jalston Fowler', 'Jimmy Graham', 'Shaquem Griffin', "Da'Ron Payne", 'Conor McDermott', 'Reshad Jones', 'Ty Sambrailo', 'A.J. Bouye', 'Kam Kelly', 'Frank Gore', 'Lawrence Timmons', 'David Fales', 'Jonathan Williams', 'Kylie Fitts', 'Jaylen Samuels', 'Fozzy Whittaker', 'Grover Stewart', 'Andre Smith', 'Jalen Tolliver', 'Johnthan Banks', 'Martinas Rankin', 'T.J. Carrie', "Da'Shawn Hand", "Za'Darius Smith", 'Nat Berhe', 'Kellen Clemens', 'Cameron Artis-Payne', 'Javon Wims', 'Terence Newman', 'Sam Bradford', 'Jonathan Woodard', 'Star Lotulelei', 'Matt Hazel', 'Jaylon Ferguson', 'Ben Niemann', 'Eddie Yarbrough', 'Carlton Davis', 'Tarell Basham', 'Grant Haley', 'Sheldon Richardson', 'Jakob Johnson', 'Breno Giacomini', 'Bruce Hector', 'Ike Boettger', 'Derek Barnett', 'Hakeem Valles', 'Kenneth Durden', 'Montez Sweat', 'Josh Allen', 'Daylon Mack', 'Nickell Robey-Coleman', 'Jeremy Maclin', 'Chidobe Awuzie', 'Marwin Evans', 'Logan Ryan', 'Tyvis Powell', 'Joey Ivie', 'Brandon Linder', 'T.J. Johnson', 'Josh Malone', 'Joshua Perry', 'Josey Jewell', 'Tyler Shatley', 'Budda Baker', 'Bobby Wagner', 'Mack Wilson', 'Ahkello Witherspoon', 'Foye Oluokun', 'Roosevelt Nix', 'Pita Taumoepenu', 'Evan Engram', 'J.D. McKissic', 'Byron Maxwell', "Ja'Wuan James", 'Jeff Wilson', 'Azeez Al-Shaair', 'Ty Nsekhe', 'Frank Zombo', 'Johnathan Cyprien', 'C.J. Mosley', 'Kasim Edebali', 'Marcus Sherels', 'Davon Godchaux', 'Isaac Yiadom', 'Josiah Tauaefa', 'Dontrelle Inman', 'Myles Jack', 'Daniel Ekuale', 'Elijah Wilkinson', 'Randall Telfer', 'Sharrod Neasman', 'Erik Swoope', 'Brian Quick', 'Terron Armstead', 'Natrell Jamerson', 'Daniel Brown', 'Donnie Ernsberger', 'Nick Chubb', 'Jeremy McNichols', 'Eric Kendricks', 'Raheem Mostert', 'J.C. Jackson', 'Brenton Bersin', 'Matthew Adams', 'Tyler Marz', 'Morgan Fox', 'Anthony Averett', 'Dion Lewis', 'Johnny Mundt', 'Tyrone Holmes', 'Janarion Grant', 'Eddie Pleasant', 'D.J. Alexander', 'Chris Conley', "Brian O'Neill", 'Jerell Adams', 'Tank Carradine', 'Brock Osweiler', 'Will Redmond', 'Doug Martin', 'Jimmy Garoppolo', 'Sean Lee', 'Ifeadi Odenigbo', 'Josh Keyes', 'Josh Johnson', 'Chandler Cox', 'Danny Woodhead', 'Chris Matthews', 'Devlin Hodges', 'Donald Payne', 'Erik McCoy', 'Alex Anzalone', 'Fred Brown', 'J.J. Nelson', 'Kemoko Turay', 'Chandon Sullivan', 'Vince Biegel', 'Ted Karras', 'Antonio Brown', 'Donovan Smith', 'Cameron Lynch', 'B.J. Bello', 'Devonta Freeman', 'Rashod Hill', 'Seth DeValve', 'Tommy Bohanon', 'Wesley Johnson', 'Lorenzo Doss', 'Kenny Moore', 'Cedric Thornton', 'Mark Nzeocha', 'Andrew Luck', 'Taylor Rapp', 'Austin Johnson', 'Trai Turner', 'DeAndre Washington', 'Mitchell Loewen', 'Kenny Golladay', 'Ryquell Armstead', 'Kevin Byard', 'Josh Wells', 'Donte Deayon', 'Adolphus Washington', 'Kingsley Keke', 'Paxton Lynch', 'Gerald Hodges', 'Chris Odom', 'Branden Jackson', 'Korey Toomer', 'Jairus Byrd', 'Corey Peters', 'David Parry', 'Ben Roethlisberger', 'Mark Barron', 'Adrian Phillips', 'Dean Marlowe', 'Korey Cunningham', 'Justin Hardy', 'Eddie Lacy', 'Anthony Nelson', 'Mason Rudolph', 'Casey Hayward', 'Eli Manning', 'Orlando Scandrick', 'Lyndon Johnson', 'Austin Seferian-Jenkins', 'Mark Andrews', 'Duane Brown', 'Eric Saubert', 'Daniel Jones', 'Marcus Davenport', 'Jamaal Charles', 'Marvin Hall', 'Case Keenum', 'Will Clarke', 'Damon Harrison', 'Lance Kendricks', 'Darren Waller', 'John Wetzel', 'Eric Tomlinson', 'Christine Michael', 'Arthur Moats', 'Ken Webster', 'Diontae Spencer', 'Kelechi Osemele', 'T.J. Clemmings', 'Jaron Brown', 'Jonathan Cooper', 'Jacoby Brissett', 'Rock Ya-Sin', 'Max McCaffrey', 'Neiko Thorpe', 'John Timu', 'Kevin Zeitler', 'Craig Robertson', 'Tyler Boyd', 'Darrelle Revis', 'Armani Watts', 'Garrett Celek', 'Maxx Crosby', 'J.J. Wilcox', 'George Johnson', 'Jared Goff', 'Kerwynn Williams', 'Ryan Mallett', 'Austin Howard', 'Jesse James', 'Spencer Pulley', 'Matt Judon', 'Dennis Daley', 'John Greco', 'Ezekiel Elliott', 'J.R. Sweezy', 'Alex Lewis', 'Cameron Fleming', 'Elijah Nkansah', 'Donte Jackson', 'Max Tuerk', 'Mike Tyson', 'Marcus Rios', 'Kenjon Barner', 'Dalvin Cook', 'Brett Hundley', 'Zack Martin', 'DeSean Jackson', 'Nigel Bradham', 'T.Y. McGill', 'Keith McGill', 'Dekoda Watson', 'Lamar Jackson', 'Kyle Murphy', 'Holton Hill', 'Chad Hansen', 'Dez Bryant', 'Jordan Devey', 'Doug Baldwin', 'Derrick Willies', 'C.J. Spiller', 'P.J. Williams', 'Kolton Miller', 'Tyquan Lewis', 'Ibraheim Campbell', 'Tyler Lockett', 'Tom Johnson', 'Denzel Ward', 'Justin Houston', 'Josh Reynolds', "Dre'Mont Jones", 'Malcolm Jenkins', 'Javien Elliott', 'Tyreek Burwell', 'Ryan Fitzpatrick', 'Luke Falk', 'Will Harris', 'Ha Ha Clinton-Dix', 'Chris Clark', 'Marcus Mariota', 'John Kuhn', 'Branden Oliver', 'Ben Jones', 'Arthur Maulet', 'Chase Roullier', 'Kaelin Clay', 'Sam Acho', 'Myles Garrett', 'Sam Young', 'Nathan Meadors', 'Nate Hairston', 'Andrew Adams', 'Rashard Higgins', 'Steven Sims', 'Jalen Myrick', 'Larry Fitzgerald', 'Justin Britt', 'Dominique Rodgers-Cromartie', 'Jordan Dangerfield', 'D.K. Metcalf', 'Ryan Ramczyk', 'Matt Moore', 'Tanner Gentry', 'Cornelius Lucas', 'A.Q. Shipley', 'Willie Henry', 'Andrew Norwell', 'Lance Dunbar', 'Damion Willis', 'Michael Johnson', 'Keelan Cole', 'Byron Pringle', 'Taquan Mizzell', 'Dak Prescott', 'Lee Smith', 'Jarrett Stidham', "Julie'n Davenport", 'Daeshon Hall', 'KhaDarel Hodge', 'Jimmy Smith', 'Kurt Coleman', 'Zach Miller', 'Ryan Kerrigan', 'Jermaine Whitehead', 'Brandon Graham', 'Torrey Smith', 'Ronald Jones', 'Demone Harris', 'Darren Fells', 'Nick Boyle', 'Ramon Foster', 'Zach Ertz', 'Matt Barkley', 'Dare Ogunbowale', 'Cornell Armstrong', 'DJ Moore', 'Xavier Cooper', 'Anthony Levine', 'Kenny Vaccaro', 'Mike Love', 'Rob Kelley', 'Rob Gronkowski', 'Taylor Lewan', 'Nick Nelson', 'Vladimir Ducasse', 'Brandon Copeland', 'John Ross', 'Harold Landry', 'Ronald Darby', 'Tramon Williams', 'Jake Martin', 'Bradley McDougald', 'Mike Glennon', 'Matthew Ioannidis', 'Jordan Williams', 'Mario Addison', 'Tyler Larsen', 'Minkah Fitzpatrick', 'Calvin Munson', 'Antonio Hamilton', 'Kyle Williams', "Pat O'Connor", 'D.J. Reader', 'Isaiah Mack', 'Andrew Beck', 'Isaiah Irving', 'Sean Spence', 'Damontre Moore', 'Tyrod Taylor', 'Eric Weems', 'Jeff Heuerman', 'Brent Grimes', 'Kwon Alexander', 'Jonathan Casillas', 'Trent Williams', 'Michael Clark', 'Allen Bailey', 'Trevon Wesco', 'Tim Settle', 'Darius Slay', 'Dallas Goedert', 'Trae Elston', 'Freddie Bishop', 'Dylan Cole', 'Derrick Jones', 'Rey Maualuga', 'Travis Swanson', 'NaVorro Bowman', 'Kyle Van Noy', 'Brian Winters', 'Shilique Calhoun', 'Maliek Collins', 'Josh Jones', 'Richie Incognito', 'Kendall Wright', 'Jeremy Lane', 'Leon Jacobs', 'Irv Smith', 'Jaydon Mickens', 'Kevin Hogan', 'Willie Young', 'Oday Aboushi', 'Quinton Jefferson', 'Austin Ekeler', 'Deon Lacey', "Hercules Mata'afa", 'Cassius Marsh', 'Anthony Steen', 'Tre Flowers', 'Leon McQuay III', 'E.J. Speed', 'Ronnie Stanley', 'Amara Darboh', "Tre'Von Johnson", 'Geneo Grissom', 'James Conner', 'Samson Ebukam', 'Ed Stinson', 'Leonard Wester', 'Xavier Woods', 'Kareem Hunt', 'Davon House', 'Scott Simonson', 'Eli Harold', 'Robert Tonyan', 'Daron Payne', 'Cody Ford', 'Rodney Hudson', 'Lane Johnson', 'Robert Thomas', "J'Mon Moore", 'Brandon Marshall', 'L.T. Walton', 'Malik Reed', 'Nick Kwiatkoski', 'David Harris', 'Joe Flacco', 'Jadeveon Clowney', 'Chris Thompson', 'Noah Brown', 'Lano Hill', 'Lewis Neal', 'Blake Cashman', 'Brynden Trawick', 'Quincy Wilson', 'Tom Savage', 'Daniel Sorensen', 'Tyson Alualu', 'Justin Simmons', 'Maxx Williams', 'Stephen Weatherly', 'Chris Hubbard', 'Bryce Treggs', 'Keith Tandy', 'Michael Gallup', 'Deandre Baker', 'Chris Harris', "Ryan O'Malley", 'Virgil Green', 'Jeff Driskel', 'Garrett Bradbury', 'Jake Matthews', 'Lenzy Pipkins', 'Tavarres King', 'Jawill Davis', 'Ray-Ray McCloud', 'Corey Levin', 'Dane Cruikshank', 'Mo Alie-Cox', 'Dre Greenlaw', 'Kirk Cousins', 'Rodger Saffold', 'Tyler Bray', 'Matt Longacre', 'Patrick Omameh', 'Spencer Ware', 'Michael Ola', 'Chandler Jones', 'Niles Scott', 'Darryl Johnson', 'Brennan Scarlett', 'Chad Wheeler', 'Cameron Sutton', 'Stephon Gilmore', 'Charles Johnson', 'Dion Dawkins', 'Chris Ivory', 'Zach Line', 'James Crawford', 'Charles Leno Jr.', 'L.J. Fort', 'Jamie Collins', 'Antone Exum', 'Hardy Nickerson', 'Alan Cross', 'Bilal Powell', 'Blidi Wreh-Wilson', 'Russell Wilson', 'Marquez Williams', 'Stevan Ridley', 'Dorance Armstrong', 'Kalen Ballage', 'Frankie Luvu', 'Ryan Hunter', 'Coby Fleener', 'Caleb Benenoch', 'Ian Silberman', 'Arie Kouandjio', 'Drue Tranquill', 'Antonio Morrison', 'Lance Lenoir', "De'Anthony Thomas", 'Bruce Irvin', 'Devaroe Lawrence', 'Darwin Thompson', 'Troymaine Pope', 'Keith Smith', 'Kam Chancellor', 'Pat Elflein', 'Trevor Williams', 'Mike Tolbert', 'Delano Hill', 'Julius Thomas', 'Brent Qvale', 'Nick Bellore', 'Victor Bolden', 'Kyzir White', 'Jonathan Allen', 'K.J. Wright', 'Anthony Sherman', 'Nick Gates', 'DeVante Parker', 'Niles Paul', 'Eddie Jackson', 'Connor Barwin', 'Rishard Matthews', 'Trent Brown', 'Terry McLaurin', 'Alex Light', 'Sebastian Joseph-Day', 'Derek Carr', 'Laremy Tunsil', 'Tarvarius Moore', 'Bilal Nichols', 'Mike Davis', 'Dexter McCoil', 'Jason Kelce', 'Marcus Kemp', 'Chris Moore', 'Keisean Nixon', 'T.Y. Hilton', 'Coty Sensabaugh', 'ArDarius Stewart', 'Kenny Young', 'Shaquil Barrett', 'Stacy McGee', 'Robert Quinn', 'C.J. Fiedorowicz', 'Kamu Grugier-Hill', 'Griff Whalen', 'Darius Jackson', 'Wesley Woodyard', 'Miles Brown', 'Darren Sproles', 'Luke Willson', 'Deadrin Senat', 'Trysten Hill', 'Jahlani Tavai', 'Xavier Williams', 'Stanley Morgan', 'Corey Nelson', 'Jordan Matthews', 'Patrick Mahomes', 'Marshall Newhouse', 'Jeff Allen', 'Efe Obada', 'Richard Ash', 'Shaun Dion Hamilton', 'Cole Beasley', 'Mychal Kendricks', 'Damion Square', 'Chunky Clements', 'Greg Robinson', 'Rodney Gunter', 'Donte Moncrief', 'David Montgomery', 'Cole Holcomb', 'Cole Wick', 'Mike Thomas', 'Al Woods', 'Kiko Alonso', 'Zane Beadles', 'Nate Allen', 'J.J. Jones', 'Taron Johnson', 'Carroll Phillips', 'Jeremy Hill', 'Josh Andrews', 'Erik Magnuson', 'Dexter Lawrence', 'Lerentee McCray', 'Anthony Brown', 'Michael Campanaro', 'Trayvon Mullen', 'Trent Harris', 'Mike Pennel', 'K.J. Brent', 'Taywan Taylor', 'Ali Marpet', 'Steve Longa', 'Tyler Matakevich', 'Denzel Perryman', 'Jullian Taylor', 'Kony Ealy', 'Kelvin Benjamin', 'Corey Robinson', 'Sylvester Williams', 'Rex Burkhead', 'Bryan Cox', 'James Daniels', 'Alfred Blue', 'Adrian Peterson', 'C.J. Ham', 'Alex Redmond', "La'el Collins", 'LeGarrette Blount', 'DeForest Buckner', 'Hunter Renfrow', 'Devin McCourty', 'Josh Norman', 'Marshawn Lynch', 'Rontez Miles', 'Scott Miller', 'Jay Elliott', 'Xavier Grimble', 'Grady Jarrett', 'Ben Braunecker', 'Eli Rogers', 'Joe Berger', 'Devin White', 'Shane Ray', 'Charles Harris', 'EJ Manuel', 'Jarran Reed', 'Greedy Williams', 'Tavon Austin', 'Taiwan Jones', 'Jordan Wilkins', 'Kalan Reed', 'Darrel Williams', 'T.J. Ward', 'Sean Smith', 'Rafael Bush', 'Connor McGovern', 'Jace Billingsley', 'Dannell Ellerbe', 'Jonotthan Harrison', 'Bryce Hager', 'Aaron Stinnie', 'Parris Campbell', 'Deon Cain', 'Alex Boone', 'Brian Robison', 'Tramaine Brock', 'Terrell Suggs', 'D.J. Reed', 'Cameron Tom', 'Kendrick Bourne', 'Mitch Morse', "D'Ernest Johnson", 'KeeSean Johnson', 'Greg Van Roten', 'Barkevious Mingo', 'Stephen Paea', 'Oshane Ximines', 'Demetrious Cox', 'Joe Barksdale', 'Eric Reid', 'Dwayne Washington', 'Eric Fisher', 'Lorenzo Carter', 'Kyle Phillips', "D'Onta Foreman", 'Adarius Glanton', 'Skai Moore', 'Darrell Henderson', 'Ryan Griffin', 'Parker Ehinger', 'Jermaine Carter', 'Jordan Leggett', 'Ricky Wagner', 'Tevin Coleman', 'Will Clapp', 'Corey Moore', 'Sean Davis', 'Brian Peters', 'Ed Eagan', 'David Sharpe', 'Anthony Johnson', 'Jonathan Jones', 'Adonis Alexander', 'Justice Hill', 'Jakobi Meyers', 'Gerald McCoy', 'Earl Mitchell', 'Darius Phillips', 'Malcom Brown', 'Marquis Flowers', 'Demario Davis', 'Trae Waynes', 'David Amerson', 'Demarcus Robinson', 'Senio Kelemete', 'Benjamin Watson', 'Ashton Dulin', 'Alex Collins', 'Ryan Switzer', 'Jaylen Hill', 'Patrick Ricard', 'Brandon Powell', 'Jaquiski Tartt', 'Vince Mayle', 'DeAndre Carter', 'Jordan Hicks', 'Bryan Witzmann', 'Henry Anderson', 'Albert Wilson', 'Nick Foles', 'Terence Garvin', 'Chris McCain', 'Andrew Wylie', 'Delanie Walker', 'Wendell Smallwood', 'Ben Jacobs', 'Jamar Taylor', 'Greg Mabin', 'Chauncey Gardner-Johnson', 'Peter Kalambayi', 'Dan Skipper', 'Charvarius Ward', "Le'Veon Bell", 'Ryan Davis', 'Javon Hargrave', 'Robert Davis', 'JC Tretter', 'Steve McLendon', 'Simeon Thomas', 'Tytus Howard', 'Christian McCaffrey', 'Elijah Lee', 'Keith Ford', 'Maurice Smith', 'Jay Prosch', 'Eric Kush', 'Sheldon Rankins', 'Darryl Roberts', 'Maurice Canady', 'Chris Landrum', 'Nick Bawden', 'Benny Cunningham', 'Dean Lowry', 'Teddy Bridgewater', 'Cam Thomas', 'Kavon Frazier', 'John Sullivan', 'Jerrell Freeman', 'Gus Edwards', 'Zach Gentry', 'Mike Adams', 'Eric Rowe', 'Johnathan Abram', 'DeAndre Hopkins', 'Charles Omenihu', 'Marlon Mack', 'Michael Dogbe', 'Jared Abbrederis', 'Jordan Leslie', 'Orson Charles', 'Isaiah Wynn', 'Cory James', 'B.W. Webb', 'Darrius Heyward-Bey', 'Quincy Williams', 'Ricky Seals-Jones', 'Devontae Booker', 'Bennie Fowler', 'Kenneth Acker', 'Gabe Wright', 'Kendall Fuller', 'Matt Forte', 'Roderic Teamer', 'Nate Solder', 'Jermon Bushrod', 'Marqise Lee', 'Khalen Saunders', 'Emmanuel Moseley', 'Beau Allen', 'Tyus Bowser', 'Derrick Morgan', 'Chad Slade', 'Ryan Glasgow', 'Brice Butler', 'Nyheim Hines', 'Brandon Scherff', 'Leighton Vander Esch', 'Jamal Carter', 'Cobi Hamilton', 'Jarrod Wilson', 'Michael Schofield', 'Carlos Watkins', 'Dominique Hatfield', 'Zach Sterup', 'Mike Hughes', "Donte' Deayon", 'Michael Burton', 'Josh Adams', 'Isaiah Oliver', 'Clive Walford', 'Justin Hollins', 'Tyrone Swoopes', 'Tajae Sharpe', 'Carl Granderson', 'LaAdrian Waddle', 'Glover Quin', 'James Carpenter', "Le'Raven Clark", 'Royce Freeman', 'Andrew Billings', 'Robby Anderson', 'Jon Bostic', 'Antonio Gates', 'Nasir Adderley', 'Fred Warner', 'Bam Bradley', 'Adam Bisnowaty', 'Kevin Pamphile', 'Jordan Poyer', 'Keenan Allen', 'Joe Mixon', 'Anthony Firkser', 'Will Holden', 'Max Scharping', 'Lardarius Webb', 'Graham Glasgow', "Hau'oli Kikaha", 'Breeland Speaks', 'Derek Watt', 'Cethan Carter', 'Tenny Palepoi', 'Cole Croston', 'Zaire Franklin', 'Shane Vereen', 'Lucas Patrick', 'Sidney Jones', 'Carson Wentz', 'Devin Singletary', 'Zay Jones', 'Tyrunn Walker', 'Bashaud Breeland', 'Dominique Easley', 'Michael Floyd', 'Nick Bosa', 'Tyrone Crawford', 'Demar Dotson', 'Cordarrelle Patterson', 'Leon Hall', 'Jalen Davis', 'Garrett Dickerson', 'Justin Hunter', 'Kyle Wilber', 'Genard Avery', 'Blake Martinez', 'Kyle Carter', 'Rashard Robinson', 'Markus Wheaton', 'Marcus Gilchrist', 'Zach Fulton', 'Corey Davis', 'Matt LaCosse', 'Chad Thomas', 'Keke Coutee', 'Roger Lewis', 'Will Richardson', 'Marcell Ateman', 'DeMarcus Walker', 'Julian Stanford', 'Maurice Hurst', 'Brittan Golden', 'Kenny Stills', 'Andre Patton', "Xavier Su'a-Filo", 'Riley Reiff', 'Chance Warmack', 'Marquis Bundy', 'Justin March-Lillard', 'Ryan Grant', 'Jerick McKinnon', 'Rhett Ellison', 'Tanoh Kpassagnon', 'Darren McFadden', 'Kalif Raymond', 'Taco Charlton', 'Taylor Gabriel', 'Brandon Tate', 'Paul Perkins', 'DeAndrew White', 'Evan Baylis', 'Tyrell Crosby', 'Lavonte David', 'Ben Banogu', 'Montravius Adams', 'Jack Doyle', 'Logan Paulsen', 'Stefen Wisniewski', 'Zeke Turner', 'Dontae Johnson', 'M.J. Stewart', 'Larry Warford', 'Bobby Massie', 'Jalen Mills', 'Elvis Dumervil', 'Sammie Coates', 'D.J. Fluker', 'Brendan Langley', 'Ereck Flowers', 'Dwayne Haskins', 'Arrelious Benn', 'Quandre Diggs', 'Tre Sullivan', 'Marqui Christian', 'A.J. Brown', 'Anthony Walker', 'C.J. Anderson', 'Trevor Siemian', 'Darryl Morris', 'Dakota Dozier', 'Geronimo Allison', 'Ced Wilson', 'Marcus Gilbert', 'Captain Munnerlyn', 'Joe Thuney', 'Andre Roberts', 'Damiere Byrd', 'Rickey Hatley', 'Jason Croom', 'Robert Ayers', 'Cody Core', 'Mike Gillislee', 'Shelton Gibson', 'Riley McCarron', 'Nathan Shepherd', 'Vinston Painter', 'Blaine Gabbert', 'Dontrell Hilliard', 'Ramik Wilson', 'Isaac Whitney', 'Leonard Williams', 'David Mayo', 'Trey Edmunds', 'Brian Orakpo', 'Jordan Franks', 'Will Fuller', 'Montae Nicholson', 'James Looney', 'Alex Armah', 'Tedric Thompson', 'Ryan Izzo', 'Ben Koyack', 'Jonathan Freeny', 'Darrell Williams', 'Muhammad Wilkerson', 'Rakeem Nunez-Roches', 'Adam Gotsis', 'Gino Gradkowski', 'Marcus Murphy', 'Julio Jones', 'Trevon Coley', 'Gareon Conley', 'Eric Weddle', 'Carlos Dunlap', 'Jeremy Kerley', 'Ben Gedeon', 'Corey Liuget', 'Amini Silatolu', 'Mecole Hardman', 'Calvin Ridley', 'Kentrell Brice', 'Maurice Alexander', 'Marcus Peters', 'David Irving', 'Chris Lammons', 'Javorius Allen', 'Andrew Brown', 'Dante Pettis', 'Devon Kennard', 'Peyton Thompson', 'Ty Montgomery', 'Tyreek Hill', 'Andrus Peat', 'Akeem Spence', 'Trent Taylor', 'Andy Levitre', 'Dwight Freeney', 'DaeSean Hamilton', 'Todd Gurley', 'Jonnu Smith', 'Nate Gerry', 'Sam Eguavoen', 'Aaron Lynch', 'Steven Means', 'Ross Travis', 'Benardrick McKinney', 'Drew Brees', 'Antwaun Woods', 'Darius Philon', 'Damien Williams', 'Phillip Supernaw', 'Jordan Reed', 'Cameron Malveaux', 'Micah Hyde', 'Stephone Anthony', 'Duke Williams', 'Ahtyba Rubin', 'Zac Kerin', 'Jaylen Watkins', 'Kamalei Correa', 'Chris Long', 'Rayshawn Jenkins', 'P.J. Hall', 'B.J. Hill', 'Ricky Ortiz', 'Alejandro Villanueva', 'Joshua Dobbs', 'Brandon Carr', 'Jared Cook', 'Jalyn Holmes', 'Renell Wren', 'Brandon Knight', 'Andre Branch', "Bene' Benwikere", 'Roy Miller', 'Zach Moore', 'Brett Jones', 'Blake Bortles', 'Nick Dzubnar', 'Denico Autry', 'Fish Smithson', 'Forrest Lamp', 'JoJo Natson', 'Brian Poole', 'Miles Boykin', 'Erik Walden', 'Ben Heeney', 'Kyle Long', 'Desmond King', 'George Iloka', 'MarQueis Gray', 'Foster Moreau', 'Kareem Jackson', 'Tavon Young', 'Vinny Curry', 'Rashad Greene', 'Aaron Jones', 'Adam Jones', 'Andre Williams', 'David King', 'Amani Hooker', 'A.J. Johnson', 'Hunter Henry', 'Marquel Lee', 'Alex Mack', 'Trevor Reilly', 'Sony Michel', 'Kendall Lamm', 'Orlando Brown', 'Nate Palmer', 'Blaine Clausell', 'Paul Posluszny', 'Terrance Mitchell', 'Joe Walker', 'Mason Foster', 'John Brown', 'Quinnen Williams', 'Xavier Rhodes', 'Tanner Vallejo', 'Bud Dupree', 'Deone Bucannon', 'Deionte Thompson', 'Darrell Daniels', 'Matt Breida', 'Aaron Wallace', 'Jakeem Grant', 'Dymonte Thomas', 'Marvin Jones', 'Chase Winovich', 'Tim Williams', 'Kevin Peterson', 'Geno Smith', 'Devin Smith', 'Ezekiel Ansah', 'Mike Gesicki', 'Deatrich Wise', 'Rashaan Gaulden', 'Terrence Fede', 'Jarvis Landry', 'Jatavis Brown', 'Morgan Burnett', 'Deiontrez Mount', 'Keenan Reynolds', 'Jaeden Graham', 'D.J. Hayden', 'Tom Kennedy', 'Obum Gwacham', 'Keith Reaser', 'Quinton Dunbar', 'Danny Johnson', 'Jarius Wright', 'Derrick Coleman', 'Chuks Okorafor', 'Jordan Roos', 'Krishawn Hogan', 'Tony McRae', "Manti Te'o", 'Matt Milano', 'Desmond Trufant', 'Elijah Qualls', 'Seth Roberts', 'Joe Jones', 'Zach Cunningham', 'Brandon Wilson', 'Tyron Smith', 'O.J. Howard', 'Jeremiah Sirles', 'Jared Veldheer', 'Sean Chandler', 'Levi Wallace', 'Neville Hewitt', 'Ugo Amadi', 'Haloti Ngata', 'Akiem Hicks', 'Alex Ellis', 'Kevin Dodd', 'Johnny Holton', 'Landon Collins', 'Josh Shaw', 'Seantrel Henderson', 'Zach Zenner', 'Juan Thornhill', 'Kelvin Harmon', 'Cameron Brate', 'Cody Latimer', 'Travis Frederick', 'Sean Weatherspoon', 'Braxton Miller', 'Carl Nassib', 'Margus Hunt', 'Christian Jones', 'John Jenkins', 'Reggie Bonnafon', 'Robert Alford', 'Tyler Kroft', 'Phillip Dorsett', 'Whitney Mercilus', 'Ryan Anderson', 'John Jerry', 'Jamison Crowder', 'Josh Rosen', 'Bisi Johnson', 'Malcolm Smith', 'Patrick Mekari', 'Jahri Evans', 'MyCole Pruitt', 'Cameron Lee', 'Kenyan Drake', 'Foley Fatukasi', 'Leonard Floyd', 'Nate Sudfeld', 'Tye Smith', 'Wayne Gallman', 'Timon Parris', 'Pete Robertson', 'Matt Flanagan', 'Zach Strief', 'Malik Jackson', 'Cameron Jordan', 'Ryan Hewitt', 'Wes Martin', 'Willie Snead', 'Jeff Janis', 'Darian Thompson', 'Demaryius Thomas', 'Stefan McClure', 'River Cracraft', 'Josh Hawkins', 'D.J. Chark', 'Braden Smith', 'Tyrell Adams', 'Devante Mays', 'Mitchell Trubisky', 'Mark Sanchez', 'Vyncint Smith', 'Gerald Everett', 'Dennis Kelly', 'Wyatt Teller', 'Alex Barrett', 'Reggie Ragland', 'Xavien Howard', 'Yannick Ngakoue', 'James White', 'Danny Vitale', 'Curtis Samuel', 'Michael Davis', 'Anthony Hitchens', 'Charles Tapper', 'Braxton Berrios', 'Alec Ogletree', 'C.J. Goodwin', 'Corey Grant', 'Lawrence Thomas', 'Mack Hollins', 'Justin Jones', 'Jerry Hughes', 'Mark Glowinski', 'Spencer Drango', 'Marcus Allen', 'Robert Nkemdiche', 'Jamaal Williams', 'Avonte Maddox', 'Cortez Broughton', 'Nick DeLuca', 'David Bakhtiari', 'Damontae Kazee', 'Jeff Holland', 'Juwann Winfree', 'Courtney Upshaw', 'John Johnson', 'Kawann Short', 'Austin Pasztor', 'Brandin Cooks', 'Jurrell Casey', 'Cody Whitehair', 'Ted Ginn', 'Kaleb McGary', 'Equanimeous St. Brown', 'Davante Adams', 'Sheldrick Redwine', 'Tom Compton', 'William Gay', 'Justin Zimmer', 'Joe Hawley', 'Greg Stroman', 'Mike McGlinchey', 'Justin Hamilton', 'Blake Jarwin', 'Eric Winston', 'Joey Bosa', 'Kemal Ishmael', 'Frostee Rucker', 'T.J. Yeldon', 'Jordan Richards', 'T.J. Watt', 'Josh Bellamy', 'Christian Kirksey', 'Tavierre Thomas', 'Alec Ingold', 'Joel Iyiegbuniwe', 'Matt Ryan', 'Terrell McClain', 'Justin McCray', 'Caleb Brantley', 'R.J. McIntosh', 'Eric Decker', 'Noah Spence', 'Marcus Johnson', 'Claude Pelon', 'Treyvon Hester', 'Jason Pierre-Paul', 'Jake Fisher', 'Ian Thomas', 'Jamil Demby', 'Ricardo Allen', 'Lane Taylor', 'Marquise Goodwin', 'Tuzar Skipper', 'Bruce Ellington', 'Matthew Stafford', 'Kyle Juszczyk', 'Kyle Love', 'Julian Edelman', 'Tavon Wilson', 'Jordan Phillips', 'Chris Carson', "K'Waun Williams", 'D.J. Moore', 'Preston Brown', 'Randall Cobb', 'DeShone Kizer', 'Mike Williams', 'Jon Feliciano', 'Saquon Barkley', 'Mark Walton', 'Russell Okung', 'Kerryon Johnson', 'Cody Kessler', 'Chuma Edoga', 'Noah Fant', 'Najee Goode', 'Jeff Cumberland', 'Karl Klug', 'Ryan Kelly', 'Christian Ringo', 'Robert Foster', 'Clayton Fejedelem', 'Matt Cassel', 'Tashawn Bower', 'Christian Wilkins', 'Tyler Patmon', 'Keishawn Bierria', 'Derek Newton', 'Tre Herndon', 'Tamba Hali', 'Joe Webb', 'Jayon Brown', 'Tyler Conklin', 'Alex Smith', 'Arik Armstead', 'Tyler Ervin', 'Justin Ellis', 'Nevin Lawson', 'Scott Tolzien', 'Siran Neal', 'Zach Vigil', 'Melvin Ingram', 'Tom Brady', 'Marquis Haynes', 'Matt Lengel', 'Ryan Bates', 'Matt Jones', 'Jameis Winston', 'Gardner Minshew', 'Jermey Parnell', 'Jamil Douglas', 'Miles Killebrew', 'Kendrick Lewis', 'Andy Janovich', 'Benny Snell', 'Golden Tate', 'Justin Davis', 'Xavier Woodson-Luster', 'Nicholas Morrow', 'Ethan Westbrooks', "Deiondre' Hall", 'Kasen Williams', 'Nathan Peterman', 'Zach Banner', 'Chris Smith', 'Bryce Petty', 'Austin Carr', 'Cordy Glenn', 'Hroniss Grasu', 'Christian Covington', 'Derron Smith', 'Jeremiah Valoaga', 'Sam Tevi', 'James Hanna', 'Curtis Riley', 'Eli Apple', 'T.J. Green', 'Trenton Scott', 'Andre Hal', 'Joe Haden', 'Philip Rivers', 'Howard Jones', 'Prince Amukamara', 'Rashaad Penny', 'Darius Kilgo', 'Halapoulivaati Vaitai', 'Jermaine Eluemunor', 'Darius Slayton', 'Jessie Bates', 'Al-Quadin Muhammad', 'Denzelle Good', 'Alterraun Verner', 'James Ferentz', 'Maurice Harris', 'Michael Bennett', 'Demarcus Lawrence', 'Tim White', 'T.J. Hockenson', 'Tremaine Edmunds', 'Destiny Vaeao', 'Adam Redmond', 'Jonathan Anderson', 'Shon Coleman', 'Robert McClain', 'Dwayne Harris', 'Brooks Reed', 'Justin Skule', 'Vincent Taylor', 'Eric Wood', 'Taylor Moton', 'Chad Beebe', 'Akeem King', 'Max Unger', 'Brian Allen', 'Tae Davis', 'Sheldon Day', 'Kenny Ladler', 'Kevin Pierre-Louis', 'Garrison Smith', 'Darius Latham', 'George Kittle', 'Tremon Smith', 'Daniel Munyer', 'Robert Griffin III', 'Daniel Kilgore', 'Danny Shelton', 'Darius Butler', 'Cory Littleton', 'Ramon Humber', 'Reggie Nelson', 'Marquez Valdes-Scantling', 'Joe Kerridge', 'Mike Purcell', 'Laquon Treadwell', 'Arthur Jones', 'Michael Brockers', 'Tion Green', 'Tyler Lancaster', 'Ryan Delaire', 'Ross Dwelley', 'Sterling Moore', 'Bronson Kaufusi', 'Trent Sherfield', 'Cam Robinson', 'Fadol Brown', 'Geno Atkins', 'Marcus Williams', 'Ty Johnson', 'Jamize Olawale', 'A.J. Klein', 'Jerel Worthy', 'Josh Gordon', 'Eric Murray', 'Dawson Knox', 'Darrius Shepherd', 'Bryson Albright', 'Rudy Ford', 'Geoff Swaim', 'Bennie Logan', 'Richard Rodgers', 'Kevin Toliver', 'Shaq Mason', 'Chris Manhertz', 'Scott Quessenberry', 'Daryl Worley', 'Kerry Hyder', 'Jerrol Garcia-Williams', 'Devin Taylor', 'William Hayes', 'Patrick Peterson', 'Martavis Bryant', 'Kyle Rudolph', 'Taylor Decker', 'Brandon Bell', "Tre'Quan Smith", 'Evan Boehm', 'Pierre Desir', 'Taylor Stallworth', 'Chris Covington', 'Joe Thomas', 'Jaelen Strong', 'Josh Robinson', 'James Cowser', 'Dion Jordan', 'Zach Pascal', 'Shaun Wilson', 'Alonzo Russell', 'Tyeler Davison', 'Ray-Ray Armstrong', 'Brent Urban', "Lil'Jordan Humphrey", 'Stephen Anderson', 'Cedric Ogbuehi', 'Matthew Slater', 'Andrew Wingard', 'Tre Boston', 'Jordan Akins', 'Shaq Thompson', 'Tony McDaniel', 'Alex Okafor', 'Lafayette Pitts', 'Wes Horton', 'D.J. Swearinger', 'Evan Smith', 'Cap Capi', 'Kenny Britt', 'Ken Crawley', 'Quincy Enunwa', 'Mike Hilton', 'Tanzel Smart', 'Johnson Bademosi', 'Joey Mbu', 'Preston Williams', 'Patrick Onwuasor', 'Clayton Geathers', 'Khari Willis', 'Henry Krieger-Coble', 'Mike Pouncey', 'Troy Reeder', 'Richard Sherman', 'Derek Wolfe', 'Bobby Rainey', 'Chuck Clark', 'Chaz Green', 'Teez Tabor', 'Brandon Bolden', 'Quinton Dial', 'Donald Penn', 'Ryan Connelly', 'Gimel President', 'Nick Easton', 'Mike Remmers', 'Shawn Lauvao', 'Haason Reddick', 'Emmanuel Lamur', 'Andre Holmes', 'David Quessenberry', 'T.J. Jones', 'Phillip Gaines', 'Darron Lee', 'Clay Matthews', 'Austin Blythe', 'Tashaun Gipson', 'Ryan Kalil', 'Johnathan Hankins', 'Nazair Jones', 'Jimmie Ward', 'Roderick Johnson', 'Marcus Epps', 'Bryan Bulaga', 'Kenny Wiggins', 'Earl Watford', 'Rodney McLeod', 'Daren Bates', 'Austin Calitro', 'Joe Looney', "Tre'Davious White", 'Deon Yelder', 'Chris Lindstrom', 'Deonte Thompson', 'Quintin Demps', 'Gunner Olszewski', 'Jacquies Smith', 'Ulrick John', 'Raven Greene', 'Brian Cushing', 'Quinten Rollins', 'William Gholston', 'Emmanuel Sanders', 'Gavin Escobar', 'Vernon Hargreaves', 'Scooby Wright', 'Jordan Willis', 'Melvin Gordon', 'A.J. Derby', 'Darnell Savage', 'Justin Watson', 'Terrell Watson', 'Steven Nelson', 'Keenan Robinson', 'E.J. Gaines', 'Jourdan Lewis', 'Austin Corbett', 'Garrett Griffin', 'Lorenzo Jerome', 'Brandon Parker', "Dorian O'Daniel", 'Jack Cichy', 'Brandon Fusco', 'Aldrick Robinson', 'Stefon Diggs', 'Cole Hikutini', 'Corey Linsley', 'Byron Murphy', 'Garry Gilliam', 'Matt Schaub', 'Joe Staley', 'Brandon Wilds', 'Allen Robinson', 'Samaje Perine', 'T.J. McDonald', 'Travis Benjamin', 'Tony Bergstrom', 'John Kelly', 'Diontae Johnson', 'Kyler Fackrell', 'Chris Wormley', 'Tony Lippett', 'Terrence Brooks', 'Dan Feeney', 'Byron Marshall', 'Pharaoh Brown', 'Paul Worrilow', 'Trevon Young', "Nick O'Leary", 'Chris Maragos', 'Vernon Hargreaves III', 'Kyler Murray', 'Ryan Russell', 'Antonio Callaway', 'Josh Perkins', 'Germain Ifedi', 'J.P. Holtz', 'David Andrews', 'Derek Rivers', 'Jarvis Jenkins', 'Chris Board', 'Damion Ratley', 'Andy Dalton', 'Josh Ferguson', 'Kyle Kalis', 'Carlos Hyde', 'Jason Verrett', 'Matt Dayes', 'Harlan Miller', 'LeSean McCoy', 'Keionta Davis', 'Josh Doctson', 'David Onyemata', 'Alan Branch', 'Kurtis Drummond', 'Parry Nickerson', 'Jason Cabinda', 'Kevin King', 'Cameron Heyward', 'Briean Boddy-Calhoun', 'Sam Hubbard', 'Roc Thomas', 'Nick Perry', 'Jack Conklin', 'Brock Coyle', 'Michael Hoomanawanui', 'Isaac Rochell', 'Bradley Marquez', 'Jerome Baker', 'Jerald Hawkins', 'Justin Lawler', 'Chester Rogers', 'Chris Johnson', 'Poona Ford', 'Dalvin Tomlinson', 'Ben Garland', 'Cyrus Kouandjio', 'Terron Ward', 'Will Beatty', 'Dawuane Smoot', 'Adrian Clayborn', 'Tanner McEvoy', 'Ifeanyi Momah', 'Terrance Williams', 'Marcus Maye', 'Joshua Garnett', 'Bobby Hart', "Da'Mari Scott", 'Dalton Risner', 'Malik Turner', 'Brandon Brooks', 'LeShaun Sims', 'Duke Shelley', 'Jamie Meder', 'Fabian Moreau', 'Jihad Ward', 'Derrick Henry', 'Nick Vigil', 'Trevor Bates', 'Carl Lawson', 'Jalen Ramsey', 'Walt Aikens', 'Adrian Colbert', 'Timmy Jernigan', 'Drew Sample', 'Jahleel Addae', 'Chad Henne', 'Rashaan Evans', 'Karl Joseph', 'Josh Jacobs', 'Keanu Neal', 'DeAndre Houston-Carson', 'David Grinnage', 'Andy Jones', 'Jay Cutler', 'Dede Westbrook', 'Khyri Thornton', 'Telvin Smith', 'Orleans Darkwa', 'Nate Davis', 'Gehrig Dieter', 'Vontaze Burfict', 'Laken Tomlinson', 'Jake Kumerow', 'Justin Currie', 'Tyrell Williams', 'Tony Jefferson', 'Derrick Kindred', 'Darius Jennings', 'Dan Arnold', 'Deebo Samuel', 'Solomon Thomas', 'Rick Wagner', 'DeMarco Murray', 'Marcus Cooper', 'Mike Hull', 'Christian Westerman', 'David Fluellen', 'Derrick Shelby', "De'Angelo Henderson Sr.", 'Drew Stanton', 'Terrance Smith', 'Kyle Fuller', 'Harvey Langi', 'Menelik Watson', "De'Lance Turner"}

JerseyNumberLabels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}

LocationLabels = {'Detroit', 'Santa Clara', 'Denver', 'Jacksonville', 'London', 'Miami Gardens', 'Houston', 'Atlanta', 'Arlington', 'Chicago', 'Tampa', 'Cleveland', 'Green Bay', 'New Orleans', 'Cincinnati', 'Philadelphia', 'East Rutherford', 'Carson', 'Charlotte', 'Seattle', 'Mexico City', 'Foxborough', 'Glendale', 'Indianapolis', 'Landover', 'Nashville', 'Orchard Park', 'Pittsburgh', 'Baltimore', 'Los Angeles', 'Minneapolis', 'Kansas City', 'Oakland'}

PlayerCollegeNameLabels = {'West Alabama', 'Syracuse', 'Newberry', 'Iowa', 'Winston-Salem State', 'Slippery Rock', 'Grambling State', 'Troy', 'East Carolina', 'Miami', 'Houston', 'Idaho', 'Western Carolina', 'USC', 'Regina, Can.', 'Coastal Carolina', 'Middle Tennessee State', 'Texas A&M', 'Malone', 'North Carolina A&T', 'Rutgers', 'New Mexico', 'Saginaw Valley State', 'Arkansas State', 'Presbyterian', 'West Texas A&M', 'Bryant', 'Brown', 'South Dakota State', 'Texas Southern', 'Miami, O.', 'Southern Mississippi', 'Florida Atlantic', 'St. Francis (PA)', 'McGill', 'Miami (Ohio)', 'Texas', 'Clemson', 'Missouri Western State', 'Furman', 'Washburn', 'Colorado', 'North Carolina', 'Boise State', 'Tennessee State', 'Connecticut', 'Southeastern Louisiana', 'Duke', 'Montana State', 'Georgia', 'Tulane', 'Middle Tennessee', 'Georgia State', 'Stillman', 'Oklahoma State', 'Washington', 'Louisiana Coll.', 'Tiffin University', 'Lindenwood', 'Stanford', 'South Dakota', 'Youngstown State', 'Concordia-St. Paul', 'Southern Methodist', 'Sacramento State', 'Kansas', 'Southeast Missouri', 'Towson', 'Mississippi State', 'Chattanooga', 'William & Mary', 'Liberty', 'Baylor', 'Louisville', 'Montana', 'Alabama State', 'California', 'Weber State', 'Alabama-Birmingham', 'Cornell', 'Vanderbilt', 'Wisconsin-Milwaukee', 'Canisius', 'Colorado State-Pueblo', 'Monmouth, N.J.', 'Louisiana State', 'Ouachita Baptist', 'Fresno State', 'Bowie State', 'Albany State, Ga.', 'Florida International', 'Pennsylvania', 'Louisiana Tech', 'Buffalo', 'Jacksonville', 'No College', 'Indiana State', 'North Alabama', 'Eastern Oregon', 'Georgia Southern', 'Delaware State', 'UCLA', 'Sam Houston State', 'Midwestern State', 'North Carolina Central', 'Minnesota', 'Illinois', 'Cincinnati', 'Alabama', 'Virginia', 'Augustana, S.D.', 'Eastern Illinois', 'Brigham Young', 'Hillsdale', 'Western State, Colo.', 'Fordham', 'Albany', 'Western Illinois', 'Bowling Green', 'Old Dominion', 'Virginia State', 'Wis.-Platteville', 'Manitoba, Can.', 'Shippensburg', 'Illinois State', 'Wisconsin-Whitewater', 'Mary Hardin-Baylor', 'Florida State', 'Nevada', 'Laval, Can.', 'Air Force', 'Ball State', 'Harvard', 'Assumption', 'San Jose State', 'Stephen F. Austin St.', 'Southern Illinois', 'Abilene Christian', 'Navy', 'Northeast Mississippi CC', 'Virginia Commonwealth', 'Southern Connecticut State', 'Central Florida', 'Michigan', 'Southern Arkansas', 'William Penn', 'Central Missouri', 'Miami (Fla.)', 'California-Irvine', 'Drake', 'South Florida', 'Arizona', 'North Carolina State', 'Rice', 'Pittsburgh', 'Western Kentucky', 'Northwest Missouri State', 'Texas-San Antonio', 'Lamar', 'Marshall', 'Memphis', 'Eastern Washington', 'Wyoming', 'Shepherd', 'Army', 'Toledo', 'Northwestern (Ia)', 'Tennessee', 'Akron', 'Greenville', 'Central Michigan', 'Bucknell', 'Fort Hays State', 'Tarleton State', 'Princeton', 'Central Arkansas', 'Utah', 'Texas-El Paso', 'Sioux Falls', 'Samford', 'Penn State', 'Western Michigan', 'McNeese State', 'LSU', 'Wagner', 'Nevada-Las Vegas', 'Southern California', 'Michigan Tech', 'Missouri Southern', 'Washington State', 'Virginia Tech', 'Tennessee-Chattanooga', 'Alcorn State', 'California-Davis', 'North Dakota State', 'Southern Utah', 'Wake Forest', 'Missouri Southern State', 'North Greenville', 'Georgia Tech', 'Ohio State', 'Stony Brook', 'Nebraska', 'Mars Hill', 'Hobart', 'Mount Union', 'Tulsa', 'Richmond', 'Northern Iowa', 'Alabama A&M', 'Columbia', 'Yale', 'Maine', 'Valdosta State', 'Oregon State', 'Arkansas-Monticello', 'Holy Cross', 'Bethune-Cookman', 'Iowa State', 'North Carolina-Charlotte', 'Delaware', 'Utah State', 'Grand Valley State', 'Wisconsin', 'Massachusetts', 'Humboldt State', 'Kansas State', 'Western Oregon', 'James Madison', 'Oklahoma', 'Eastern Michigan', 'Concordia, Minn', 'Monmouth (N.J.)', 'Missouri', 'Azusa Pacific', 'Temple', 'South Carolina State', 'Bloomsburg', 'Arkansas-Pine Bluff', 'Limestone', 'Florida', 'Kentucky', 'Incarnate Word (Tex.)', 'Texas Christian', 'Grambling', 'Kent State', 'San Diego', 'Ferris State', 'Arkansas', 'East Central', 'Northern Illinois', 'West Virginia', 'Hawaii', 'Auburn', 'Notre Dame', 'Colorado State', 'Indiana', 'Beloit', 'Boston College', 'Oregon', 'Arizona State', 'Bowling Green State', 'Frostburg State', 'Nebraska-Omaha', 'Pittsburg State', 'Cal Poly', 'Minn. State-Mankato', 'Missouri State', 'South Alabama', 'Michigan State', 'Portland State', 'North Texas', 'New Mexico State', 'Henderson State', 'Howard', 'New Hampshire', 'Texas State', 'Citadel', 'California, Pa.', 'Wofford', 'Villanova', 'Charleston, W. Va.', 'South Carolina', 'Mississippi', 'West Georgia', 'Ashland', 'Northwestern State-Louisiana', 'Nicholls State', 'San Diego State', 'Idaho State', 'Ohio U.', 'Louisiana-Lafayette', 'Marist', 'Bemidji State', 'Ohio', 'Belhaven', 'Kentucky Wesleyan', 'Marian (Ind.)', 'Maryland', 'Northwestern', 'Purdue', 'Eastern Kentucky', 'Hampton', 'Jacksonville State', 'Appalachian State', 'Stetson', 'Texas Tech', 'Murray State', 'Prairie View', 'Southern University', 'Chadron State'}

StadiumLabels = {'Gillette Stadium', 'Mercedes-Benz Dome', 'Arrowhead Stadium', 'NRG Stadium', 'Twickenham Stadium', 'Mercedes-Benz Superdome', 'Broncos Stadium At Mile High', 'Tottenham Hotspur Stadium', 'CenturyLink', 'First Energy Stadium', 'Raymond James Stadium', 'Bank of America Stadium', 'FirstEnergy Stadium', 'Lambeau Field', 'Soldier Field', 'M & T Bank Stadium', 'Everbank Field', 'Estadio Azteca', 'State Farm Stadium', 'Ford Field', 'FedExField', 'M&T Stadium', 'Lambeau field', 'FedexField', 'FirstEnergyStadium', 'AT&T Stadium', 'TIAA Bank Field', 'Lucas Oil Stadium', 'Levis Stadium', 'NRG', 'Lincoln Financial Field', 'FirstEnergy', 'Los Angeles Memorial Coliesum', 'Wembley Stadium', 'University of Phoenix Stadium', 'Paul Brown Stdium', 'Los Angeles Memorial Coliseum', 'U.S. Bank Stadium', 'Heinz Field', 'Mercedes-Benz Stadium', 'Oakland Alameda-County Coliseum', 'MetLife', 'StubHub Center', 'Oakland-Alameda County Coliseum', 'CenturyLink Field', 'CenturyField', 'Sports Authority Field at Mile High', 'Nissan Stadium', 'Metlife Stadium', 'Twickenham', 'Dignity Health Sports Park', 'Empower Field at Mile High', 'Paul Brown Stadium', 'MetLife Stadium', 'New Era Field', 'Broncos Stadium at Mile High', 'M&T Bank Stadium', 'EverBank Field', 'Hard Rock Stadium'}

StadiumTypeLabels = {'Heinz Field', 'Indoor, Roof Closed', 'Open', 'Domed, closed', 'Outdoor', 'Retr. Roof-Closed', 'Indoor, Open Roof', 'Bowl', 'Cloudy', 'Outdoor Retr Roof-Open', 'Dome', 'Indoor', 'Domed, Open', 'Retr. Roof-Open', 'Retractable Roof'}

TurfLabels = {'Natural grass', 'UBU Sports Speed S5-M', 'Field Turf', 'SISGrass', 'Artifical', 'Twenty Four/Seven Turf', 'Turf', 'A-Turf Titan', 'FieldTurf 360', 'UBU-Speed Series-S5-M', 'Natural', 'DD GrassMaster', 'Grass'}

OffenseFormationLabels = {'WILDCAT', 'I_FORM', 'EMPTY', 'SHOTGUN', 'ACE', 'PISTOL', 'SINGLEBACK', 'JUMBO'}

HomeTeamAbbrLabels = {'NYG', 'CLE', 'HOU', 'JAX', 'NO', 'DET', 'IND', 'LAC', 'TB', 'SF', 'PHI', 'ATL', 'DAL', 'NE', 'BUF', 'CHI', 'MIA', 'ARI', 'TEN', 'LA', 'KC', 'BAL', 'CIN', 'OAK', 'PIT', 'DEN', 'SEA', 'NYJ', 'WAS', 'GB', 'MIN', 'CAR'}

VisitorTeamAbbrLabels = {'NYG', 'CLE', 'HOU', 'JAX', 'NO', 'DET', 'IND', 'LAC', 'TB', 'SF', 'PHI', 'ATL', 'NE', 'DAL', 'BUF', 'CHI', 'MIA', 'ARI', 'TEN', 'LA', 'KC', 'BAL', 'OAK', 'CIN', 'PIT', 'DEN', 'SEA', 'NYJ', 'WAS', 'GB', 'MIN', 'CAR'}

# drop GameWeather
# # Before Label Encode

# DisplayNames,JerseyNumbers = list(df_train.DisplayName.unique()),list(df_train.JerseyNumber.unique())

# Locations,PlayerCollegeNames = list(df_train.Location.unique()),list(df_train.PlayerCollegeName.unique())

# Stadiums,StadiumTypes,Turfs = list(df_train.Stadium.unique()),list(df_train.StadiumType.unique()),list(df_train.Turf.unique())

# HomeTeamAbbrs,VisitorTeamAbbrs = list(df_train.HomeTeamAbbr.unique()),list(df_train.VisitorTeamAbbr.unique())

# GameWeathers = list(df_train.GameWeather.unique())

# OffenseFormations = list(df_train.OffenseFormation.unique())
from sklearn.preprocessing import LabelEncoder



label_columns = {

    'DisplayName':DisplayNameLabels,'JerseyNumber':JerseyNumberLabels,'Location':LocationLabels,'PlayerCollegeName':PlayerCollegeNameLabels,

    'Stadium':StadiumLabels,'StadiumType':StadiumTypeLabels,'Turf':TurfLabels,'OffenseFormation':OffenseFormationLabels,'HomeTeamAbbr':HomeTeamAbbrLabels,

    'VisitorTeamAbbr':VisitorTeamAbbrLabels}

LES = {}

for col in label_columns.keys():

    LES[col] = LabelEncoder().fit(list(label_columns[col]))

    df_train[col] = LES[col].transform(df_train[col])

    

LES_2 = {}

for col in df_train.columns:

    if df_train[col].dtype == 'object':

        LES_2[col] = LabelEncoder().fit(df_train[col])

        df_train[col] = LES_2[col].transform(df_train[col])
# print(df_train.Location.unique())

# print('*'*100)

# print(LES['Location'].classes_)
df_train.info()
df_train = df_train.drop(['GameClock','TimeHandoff','TimeSnap','GameWeather'], axis=1)
%%time

aggregation_columns = {}

new_cols = ['S_mean_offense','A_mean_offense','Height_mean_offense','Weight_mean_offense','Age_mean_offense',

           'S_mean_defense','A_mean_defense','Height_mean_defense','Weight_mean_defense','Age_mean_defense',

           'offense_mean_diss','defense_mean_diss',

           'offense_mean_diss_laghalf','defense_mean_diss_laghalf','offense_mean_diss_lagone','defense_mean_diss_lagone',

           'offense_mean_diss_lag2','defense_mean_diss_lag2','offense_mean_diss_lag3','defense_mean_diss_lag3',

           'defender_count_yard3','defender_count_yard5','defender_count_yard3_lag1','defender_count_yard5_lag1',

           'defender_count_yard3_lag2','defender_count_yard5_lag2','defender_count_yard3_lag3','defender_count_yard5_lag3',

           'offenser_count_yard3','offenser_count_yard5','offenser_count_yard3_lag1','offenser_count_yard5_lag1',

           'offenser_count_yard3_lag2','offenser_count_yard5_lag2','offenser_count_yard3_lag3','offenser_count_yard5_lag3',]



for col in list(df_train.columns)+new_cols:

    aggregation_columns[col] = []

for need_del_col in ['GameId','PlayId','NflId','NflIdRusher','PossessionTeam','Offense']:

    del aggregation_columns[need_del_col]



df_aggregation = pd.DataFrame(columns=aggregation_columns)



for k,chance in df_train.groupby(['GameId','PlayId']):

    game_id,play_id = gp[0],gp[1]

    # S=vt+1/2*at^2

    chance['X_lag_half'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*.5+.5*l.A*(.5**2))*np.sin(l.Dir), axis=1)

    chance['X_lag_1'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*1+.5*l.A*(1**2))*np.sin(l.Dir), axis=1)

    chance['X_lag_2'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*2+.5*l.A*(2**2))*np.sin(l.Dir), axis=1)

    chance['X_lag_3'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*3+.5*l.A*(3**2))*np.sin(l.Dir), axis=1)

    chance['Y_lag_half'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*.5+.5*l.A*(.5**2))*np.sin(l.Dir), axis=1)

    chance['Y_lag_1'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*1+.5*l.A*(1**2))*np.sin(l.Dir), axis=1)

    chance['Y_lag_2'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*2+.5*l.A*(2**2))*np.sin(l.Dir), axis=1)

    chance['Y_lag_3'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*3+.5*l.A*(3**2))*np.sin(l.Dir), axis=1)

    rusher = chance[chance.NflId==chance.NflIdRusher]

    offense = chance[chance.Offense]

    defense = chance[~chance.Offense]



    # 聚合处理：

    chance_series = rusher.drop(['GameId','PlayId','NflId','NflIdRusher','PossessionTeam','Offense'], axis=1).iloc[0]

    rusher = rusher.iloc[0]

    

    # 静态信息

    chance_series['S_mean_offense'] = offense.S.mean()

    chance_series['A_mean_offense'] = offense.A.mean()

    chance_series['Height_mean_offense'] = offense.PlayerHeight.mean()

    chance_series['Weight_mean_offense'] = offense.PlayerWeight.mean()

    chance_series['Age_mean_offense'] = offense.Age.mean()

    chance_series['S_mean_defense'] = defense.S.mean()

    chance_series['A_mean_defense'] = defense.A.mean()

    chance_series['Height_mean_defense'] = defense.PlayerHeight.mean()

    chance_series['Weight_mean_defense'] = defense.PlayerWeight.mean()

    chance_series['Age_mean_defense'] = defense.Age.mean()

    

    # XYAS挖掘

    # 提取：目前针对的计算不同，有针对offense的，有针对defense的

    # np.linalg.norm(array(x,y)-array(x,y)) 求点的欧氏距离

    def xyas(row):

        # 0,0.5,1,2,3内各个球员与持球人的距离

        distance_now = np.linalg.norm(np.array([row.X,row.Y])-np.array([rusher.X,rusher.Y]))

        distance_laghalf = np.linalg.norm(np.array([row.X_lag_half,row.Y_lag_half])-np.array([rusher.X_lag_half,rusher.Y_lag_half]))

        distance_lag1 = np.linalg.norm(np.array([row.X_lag_1,row.Y_lag_1])-np.array([rusher.X_lag_1,rusher.Y_lag_1]))

        distance_lag2 = np.linalg.norm(np.array([row.X_lag_2,row.Y_lag_2])-np.array([rusher.X_lag_2,rusher.Y_lag_2]))

        distance_lag3 = np.linalg.norm(np.array([row.X_lag_3,row.Y_lag_3])-np.array([rusher.X_lag_3,rusher.Y_lag_3]))

        # 判断对应时间段后是否小于3码、5码

        return pd.Series([distance_now,distance_laghalf,distance_lag1,distance_lag2,distance_lag3,

                            distance_now<=3,distance_now<=5,distance_laghalf<=3,distance_laghalf<=5,

                            distance_lag1<=3,distance_lag1<=5,distance_lag2<=3,distance_lag2<=5,

                            distance_lag3<=3,distance_lag3<=5], 

                            index=['dis0','dish','dis1','dis2','dis3','dis0_3','dis0_5','dish_3','dish_5','dis1_3','dis1_5','dis2_3','dis2_5','dis3_3','dis3_5'])

    

    offense_xyas = offense[offense.NflId!=offense.NflIdRusher].apply(xyas, axis=1)

    defense_xyas = defense.apply(xyas, axis=1)

    

    chance_series['offense_mean_diss'] = (offense_xyas['dis0'].mean())

    chance_series['defense_mean_diss'] = (defense_xyas['dis0'].mean())

    chance_series['offense_mean_diss_laghalf'] = (offense_xyas['dish'].mean())

    chance_series['defense_mean_diss_laghalf'] = (defense_xyas['dish'].mean())

    chance_series['offense_mean_diss_lagone'] = (offense_xyas['dis1'].mean())

    chance_series['defense_mean_diss_lagone'] = (defense_xyas['dis1'].mean())

    chance_series['offense_mean_diss_lag2'] = (offense_xyas['dis2'].mean())

    chance_series['defense_mean_diss_lag2'] = (defense_xyas['dis2'].mean())

    chance_series['offense_mean_diss_lag3'] = (offense_xyas['dis3'].mean())

    chance_series['defense_mean_diss_lag3'] = (defense_xyas['dis3'].mean())

    

    chance_series['defender_count_yard3'] = (len(defense_xyas[defense_xyas['dis0_3']]))

    chance_series['defender_count_yard3_lag1'] = (len(defense_xyas[defense_xyas['dis1_3']]))

    chance_series['defender_count_yard3_lag2'] = (len(defense_xyas[defense_xyas['dis2_3']]))

    chance_series['defender_count_yard3_lag3'] = (len(defense_xyas[defense_xyas['dis3_3']]))

    chance_series['defender_count_yard5'] = (len(defense_xyas[defense_xyas['dis0_5']]))

    chance_series['defender_count_yard5_lag1'] = (len(defense_xyas[defense_xyas['dis1_5']]))

    chance_series['defender_count_yard5_lag2'] = (len(defense_xyas[defense_xyas['dis2_5']]))

    chance_series['defender_count_yard5_lag3'] = (len(defense_xyas[defense_xyas['dis3_5']]))

    

    chance_series['offenser_count_yard3'] = (len(offense_xyas[offense_xyas['dis0_3']]))

    chance_series['offenser_count_yard3_lag1'] = (len(offense_xyas[offense_xyas['dis1_3']]))

    chance_series['offenser_count_yard3_lag2'] = (len(offense_xyas[offense_xyas['dis2_3']]))

    chance_series['offenser_count_yard3_lag3'] = (len(offense_xyas[offense_xyas['dis3_3']]))

    chance_series['offenser_count_yard5'] = (len(offense_xyas[offense_xyas['dis0_5']]))

    chance_series['offenser_count_yard5_lag1'] = (len(offense_xyas[offense_xyas['dis1_5']]))

    chance_series['offenser_count_yard5_lag2'] = (len(offense_xyas[offense_xyas['dis2_5']]))

    chance_series['offenser_count_yard5_lag3'] = (len(offense_xyas[offense_xyas['dis3_5']]))

    

    # 全局信息

    

    # 其他信息

    

    df_aggregation = df_aggregation.append(chance_series, ignore_index=True)

    

for col in df_aggregation.columns:

    if df_aggregation[col].dtype == 'object':

        df_aggregation[col] = df_aggregation[col].astype('int')
# 1/0
train_columns = list(df_aggregation.columns).copy()

train_columns.remove('Yards')
model = RegressorConditional()

model.fit(df_aggregation[train_columns], df_aggregation.Yards)



plt.figure(figsize=(12, 4))

for oc in model.dist:

    plt.plot(model.dist[oc], label=oc)

plt.xticks(list(range(-1, 200, 25)), list(range(-100, 101, 25)))

plt.legend()

plt.show()
tmp_test = []

test__ = None
# for tmp_test_ in tmp_test:

#     tmp = tmp_test_.copy()

#     test_process(tmp)

#     break
def test_process(df_test):

    # 数据类型对齐

    print('数据类型对齐')

    df_test.TimeHandoff=df_test.TimeHandoff.astype('datetime64')

    df_test.TimeSnap=df_test.TimeSnap.astype('datetime64')

    

    # 填充

    print('缺失填充')

    df_test[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']] = df_test[['GameWeather','Temperature','Humidity','WindSpeed','WindDirection']].fillna(method='ffill')



    def fill_stadiumtype(row):

        if row['Stadium'] in ['StubHub Center','MetLife Stadium'] and pd.isnull(row['StadiumType']):

            return 'Outdoor'

        return row['StadiumType']



    df_test.StadiumType = df_test.apply(fill_stadiumtype, axis=1)

    df_test.StadiumType = df_test.StadiumType.fillna(method='ffill')



    df_test.FieldPosition = df_test.FieldPosition.fillna('Middle')



    df_test.OffenseFormation = df_test.OffenseFormation.fillna('SINGLEBACK')



    def fill_defendersinthebox(row):

        if pd.isnull(row['DefendersInTheBox']):

            return defendersInTheBox[row['Team']][row['HomeTeamAbbr']][row['VisitorTeamAbbr']][row['DefensePersonnel']]

        return row['DefendersInTheBox']



    df_test.DefendersInTheBox = df_test.apply(fill_defendersinthebox, axis=1)



    df_test.Orientation = df_test.Orientation.fillna(df_test.Orientation.mean())

    df_test.Dir = df_test.Dir.fillna(df_test.Dir.mean())



    # 异常、重复

    print('异常、重复处理')

    df_test.StadiumType = df_test.StadiumType.map(stadiumtype_map)

    df_test.PossessionTeam = df_test.PossessionTeam.apply(lambda pt:possessionteam_map[pt] if pt in possessionteam_map.keys() else pt)

    df_test.Location = df_test.Location.apply(lambda pt:location_map[pt] if pt in location_map.keys() else pt)

    df_test.Turf = df_test.Turf.apply(lambda pt:turf_map[pt] if pt in turf_map.keys() else pt)



    # EDA

    print('EDA')

    df_test['TeamBelongAbbr'] = df_test.apply(lambda row:row['HomeTeamAbbr'] if row['Team']=='home' else row['VisitorTeamAbbr'],axis=1)

    df_test['Offense'] = df_test.apply(lambda row:row['PossessionTeam']==row['TeamBelongAbbr'],axis=1)



    # FE

    print('FE')

    #df_test = df_test.drop(['DisplayName','JerseyNumber','WindSpeed','WindDirection'], axis=1)

    df_test = df_test.drop(['WindSpeed','WindDirection'], axis=1)

#     df_test.PossessionTeam = df_test.apply(lambda row:1 if row['PossessionTeam']==row['TeamBelongAbbr'] else 0, axis=1)

#     df_test.FieldPosition = df_test.apply(lambda row:1 if row['FieldPosition']==row['TeamBelongAbbr'] else 0, axis=1)

    df_test.DefendersInTheBox = df_test.DefendersInTheBox.astype('int8')

    df_test.PlayerHeight = df_test.PlayerHeight.apply(lambda height:int(height[0])*12+int(height[2:])).astype('int')

    df_test['Age'] = df_test.PlayerBirthDate.apply(lambda bd:2019-int(bd[-4:]))

    df_test = df_test.drop(['PlayerBirthDate'], axis=1)



    df_test['TimeFromSnapToHandoff'] = (df_test.TimeHandoff - df_test.TimeSnap).apply(lambda x:x.total_seconds()).astype('int8')



    df_test['GameDuration'] = (df_test.GameClock.apply(lambda gc:15*60-int(gc[:2])*60-int(gc[3:5]))) + (df_test.Quarter-1)*15*60

    

    for POS in POSITIONS:

        df_test['Position_'+POS] = df_test[['OffensePersonnel','DefensePersonnel']].apply(split_pos,args=(POS,),axis=1)

    df_test = df_test.drop(['OffensePersonnel','DefensePersonnel'], axis=1)

    

    df_test['GoalZone'] = df_test[['FieldPosition','TeamBelongAbbr','YardLine']].apply(lambda pty:1 if pty['FieldPosition']!=pty['TeamBelongAbbr'] and pty['YardLine']<=10 else 0, axis=1)

    

    df_test['FirstDownDanger'] = df_test[['Distance','Down']].apply(lambda dd:1 if dd['Down']>3 and dd['Distance']>5 else 0, axis=1)

    

    df_test['DistanceTouchDown'] = df_test[['YardLine','FieldPosition','PossessionTeam']].apply(lambda yfp:100-yfp['YardLine'] if(yfp['PossessionTeam']==yfp['FieldPosition']) else yfp['YardLine'], axis=1)

    

    # 测试数据对字符串类型做兜底fillna

    for col in (df_test.columns & df_train.columns):

        if df_test[col].dtype == 'object' and df_test[col].isnull().sum()>0:

            df_test[col] = df_test[col].fillna(object_columns[col])

    

    # zzz



#     global DisplayNames,JerseyNumbers,Locations,PlayerCollegeNames,Stadiums,StadiumTypes,Turfs,HomeTeamAbbrs,VisitorTeamAbbrs,GameWeathers,OffenseFormations

#     DisplayNames += list(df_test.DisplayName.unique())

#     JerseyNumbers += list(df_test.JerseyNumber.unique())

#     Locations += list(df_test.Location.unique())

#     PlayerCollegeNames += list(df_test.PlayerCollegeName.unique())

#     Stadiums += list(df_test.Stadium.unique())

#     StadiumTypes += list(df_test.StadiumType.unique())

#     Turfs += list(df_test.Turf.unique())

#     HomeTeamAbbrs += list(df_test.HomeTeamAbbr.unique())

#     VisitorTeamAbbrs += list(df_test.VisitorTeamAbbr.unique())

#     GameWeathers += list(df_test.GameWeather.unique())

#     OffenseFormations += list(df_test.OffenseFormation.unique())



    # xxx



#     df_test.DisplayName = 1

#     df_test.JerseyNumber = 1

#     df_test.Location = 1

#     df_test.PlayerCollegeName = 1

#     df_test.Stadium = 1

#     df_test.StadiumType = 1

#     df_test.Turf = 1

#     df_test.GameWeather = 1

#     df_test.OffenseFormation = 1

#     df_test.HomeTeamAbbr = 1

#     df_test.VisitorTeamAbbr = 1

    

    df_test = df_test.drop(['GameClock','TimeHandoff','TimeSnap','GameWeather'], axis=1)

    

    for col in label_columns.keys():

        df_test[col] = LES[col].transform(df_test[col])

    for col in df_test.columns:

        if df_test[col].dtype == 'object':

            df_test[col] = LES_2[col].transform(df_test[col])

    

    # 避免测试数据与训练数据不一致导致的缺失问题无法被处理到

    print('健壮兜底处理')

    df_test = df_test.fillna(-999)



    # 聚合

    print('聚合')

    aggregation_columns_test = {}

    for col in list(df_test.columns)+new_cols:

        aggregation_columns_test[col] = []

    for need_del_col in ['GameId','PlayId','NflId','NflIdRusher','PossessionTeam','Offense']:

        del aggregation_columns_test[need_del_col]



    df_aggregation_test = pd.DataFrame(columns=aggregation_columns_test)

    for k,chance in df_test.groupby(['GameId','PlayId']):

        game_id,play_id = gp[0],gp[1]

        # S=vt+1/2*at^2

        chance['X_lag_half'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*.5+.5*l.A*(.5**2))*np.sin(l.Dir), axis=1)

        chance['X_lag_1'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*1+.5*l.A*(1**2))*np.sin(l.Dir), axis=1)

        chance['X_lag_2'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*2+.5*l.A*(2**2))*np.sin(l.Dir), axis=1)

        chance['X_lag_3'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.X+(l.S*3+.5*l.A*(3**2))*np.sin(l.Dir), axis=1)

        chance['Y_lag_half'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*.5+.5*l.A*(.5**2))*np.sin(l.Dir), axis=1)

        chance['Y_lag_1'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*1+.5*l.A*(1**2))*np.sin(l.Dir), axis=1)

        chance['Y_lag_2'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*2+.5*l.A*(2**2))*np.sin(l.Dir), axis=1)

        chance['Y_lag_3'] = chance[['X','Y','S','A','Dir']].apply(lambda l:l.Y+(l.S*3+.5*l.A*(3**2))*np.sin(l.Dir), axis=1)

        rusher = chance[chance.NflId==chance.NflIdRusher]

        offense = chance[chance.Offense]

        defense = chance[~chance.Offense]

        

        chance_series = rusher.drop(['GameId','PlayId','NflId','NflIdRusher','PossessionTeam','Offense'], axis=1).iloc[0]

        rusher = rusher.iloc[0]

        

        # 球员信息

        chance_series['S_mean_offense'] = chance[chance.Offense==1].S.mean()

        chance_series['A_mean_offense'] = chance[chance.Offense==1].A.mean()

        chance_series['Height_mean_offense'] = chance[chance.Offense==1].PlayerHeight.mean()

        chance_series['Weight_mean_offense'] = chance[chance.Offense==1].PlayerWeight.mean()

        chance_series['Age_mean_offense'] = chance[chance.Offense==1].Age.mean()

        chance_series['S_mean_defense'] = chance[chance.Offense==0].S.mean()

        chance_series['A_mean_defense'] = chance[chance.Offense==0].A.mean()

        chance_series['Height_mean_defense'] = chance[chance.Offense==0].PlayerHeight.mean()

        chance_series['Weight_mean_defense'] = chance[chance.Offense==0].PlayerWeight.mean()

        chance_series['Age_mean_defense'] = chance[chance.Offense==0].Age.mean()

        

        # XYAS挖掘

        # 提取：目前针对的计算不同，有针对offense的，有针对defense的

        # np.linalg.norm(array(x,y)-array(x,y)) 求点的欧氏距离

        def xyas(row):

            # 0,0.5,1,2,3内各个球员与持球人的距离

            distance_now = np.linalg.norm(np.array([row.X,row.Y])-np.array([rusher.X,rusher.Y]))

            distance_laghalf = np.linalg.norm(np.array([row.X_lag_half,row.Y_lag_half])-np.array([rusher.X_lag_half,rusher.Y_lag_half]))

            distance_lag1 = np.linalg.norm(np.array([row.X_lag_1,row.Y_lag_1])-np.array([rusher.X_lag_1,rusher.Y_lag_1]))

            distance_lag2 = np.linalg.norm(np.array([row.X_lag_2,row.Y_lag_2])-np.array([rusher.X_lag_2,rusher.Y_lag_2]))

            distance_lag3 = np.linalg.norm(np.array([row.X_lag_3,row.Y_lag_3])-np.array([rusher.X_lag_3,rusher.Y_lag_3]))

            # 判断对应时间段后是否小于3码、5码

            return pd.Series([distance_now,distance_laghalf,distance_lag1,distance_lag2,distance_lag3,

                                distance_now<=3,distance_now<=5,distance_laghalf<=3,distance_laghalf<=5,

                                distance_lag1<=3,distance_lag1<=5,distance_lag2<=3,distance_lag2<=5,

                                distance_lag3<=3,distance_lag3<=5], 

                                index=['dis0','dish','dis1','dis2','dis3','dis0_3','dis0_5','dish_3','dish_5','dis1_3','dis1_5','dis2_3','dis2_5','dis3_3','dis3_5'])



        offense_xyas = offense[offense.NflId!=offense.NflIdRusher].apply(xyas, axis=1)

        defense_xyas = defense.apply(xyas, axis=1)



        chance_series['offense_mean_diss'] = (offense_xyas['dis0'].mean())

        chance_series['defense_mean_diss'] = (defense_xyas['dis0'].mean())

        chance_series['offense_mean_diss_laghalf'] = (offense_xyas['dish'].mean())

        chance_series['defense_mean_diss_laghalf'] = (defense_xyas['dish'].mean())

        chance_series['offense_mean_diss_lagone'] = (offense_xyas['dis1'].mean())

        chance_series['defense_mean_diss_lagone'] = (defense_xyas['dis1'].mean())

        chance_series['offense_mean_diss_lag2'] = (offense_xyas['dis2'].mean())

        chance_series['defense_mean_diss_lag2'] = (defense_xyas['dis2'].mean())

        chance_series['offense_mean_diss_lag3'] = (offense_xyas['dis3'].mean())

        chance_series['defense_mean_diss_lag3'] = (defense_xyas['dis3'].mean())



        chance_series['defender_count_yard3'] = (len(defense_xyas[defense_xyas['dis0_3']]))

        chance_series['defender_count_yard3_lag1'] = (len(defense_xyas[defense_xyas['dis1_3']]))

        chance_series['defender_count_yard3_lag2'] = (len(defense_xyas[defense_xyas['dis2_3']]))

        chance_series['defender_count_yard3_lag3'] = (len(defense_xyas[defense_xyas['dis3_3']]))

        chance_series['defender_count_yard5'] = (len(defense_xyas[defense_xyas['dis0_5']]))

        chance_series['defender_count_yard5_lag1'] = (len(defense_xyas[defense_xyas['dis1_5']]))

        chance_series['defender_count_yard5_lag2'] = (len(defense_xyas[defense_xyas['dis2_5']]))

        chance_series['defender_count_yard5_lag3'] = (len(defense_xyas[defense_xyas['dis3_5']]))



        chance_series['offenser_count_yard3'] = (len(offense_xyas[offense_xyas['dis0_3']]))

        chance_series['offenser_count_yard3_lag1'] = (len(offense_xyas[offense_xyas['dis1_3']]))

        chance_series['offenser_count_yard3_lag2'] = (len(offense_xyas[offense_xyas['dis2_3']]))

        chance_series['offenser_count_yard3_lag3'] = (len(offense_xyas[offense_xyas['dis3_3']]))

        chance_series['offenser_count_yard5'] = (len(offense_xyas[offense_xyas['dis0_5']]))

        chance_series['offenser_count_yard5_lag1'] = (len(offense_xyas[offense_xyas['dis1_5']]))

        chance_series['offenser_count_yard5_lag2'] = (len(offense_xyas[offense_xyas['dis2_5']]))

        chance_series['offenser_count_yard5_lag3'] = (len(offense_xyas[offense_xyas['dis3_5']]))

        

        df_aggregation_test = df_aggregation_test.append(chance_series, ignore_index=True)

    

    return df_aggregation_test
%%time

from kaggle.competitions import nflrush



names = dict(zip(range(199), ['Yards%d' % i for i in range(-99, 100)]))



env = nflrush.make_env()

for df_test, _ in env.iter_test():

    tmp_test.append(df_test.copy())

    df_test = test_process(df_test)

    env.predict(pd.DataFrame([np.clip(np.cumsum(model.predict_proba(df_test)), 0, 1)]).rename(names, axis=1))

env.write_submission_file()
# last_tmp = tmp_test[-1]

# last_tmp[label_columns]