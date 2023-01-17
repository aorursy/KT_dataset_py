	

import glob

import pandas

import pandasql

 

def find_file():

    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

 

def run_sql(query):

    return pandasql.sqldf(query, globals())

 

Schools = pandas.read_excel(find_file(), sheet_name="Schools")

print(Schools)



Games = pandas.read_excel(find_file(), sheet_name="Games")

print(Games)



Accounts = pandas.read_excel(find_file(), sheet_name="Accounts")

print(Accounts)



Posts = pandas.read_excel(find_file(), sheet_name="Posts")

print(Posts)

 

HomeWins = run_sql("""

    select *

    from Games

    where GameHomeScore > GameAwayScore

""")

 

print(HomeWins)



AwayWins = run_sql("""

    select *

    from Games

    where GameHomeScore < GameAwayScore

""")

 

print(AwayWins)



Ties = run_sql("""

    select *

    from Games

    where GameHomeScore = GameAwayScore

""")

 

print(Ties)