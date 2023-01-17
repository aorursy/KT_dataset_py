import glob

import pandas

import pandasql

 

def find_file():

    return glob.glob("../input/**/*.xlsx", recursive=True)[0]

 

def run_sql(query):

    return pandasql.sqldf(query, globals())



Distance = pandas.read_excel(find_file(),sheet_name="Distance")

print(Distance)



Race = pandas.read_excel(find_file(),sheet_name="Race")

print(Race)



Location = pandas.read_excel(find_file(),sheet_name="Location")

print(Location)



Athlete = pandas.read_excel(find_file(),sheet_name="Athlete")

print(Athlete)



RaceAthlete = pandas.read_excel(find_file(),sheet_name="RaceAthlete")

print(RaceAthlete)



sprints = run_sql("""

    select *

    from Race

    where DistanceID=1

    """)

print(sprints)



locations = run_sql("""

    select *

    from Race

    where LocationID=2

""")

print(locations)



athlete = run_sql("""

    select Athlete.Name, Race.RaceName

    from Race, RaceAthlete, Athlete

    where Athlete.AthleteID=RaceAthlete.AthleteID and Race.RaceName="Door County Triathlon"

""")

print(athlete)



race = run_sql("""

    select RaceName

    from Race

    where RaceID=2

""")