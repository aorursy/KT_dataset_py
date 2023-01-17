!pip install nameparser
import pandas as pd
import json
from nameparser import HumanName
import requests
from io import BytesIO
# Import the file, replace NaN with blanks
astro_google_sheets_url  = r'https://docs.google.com/spreadsheets/d/e/2PACX-1vRTROflYTQISuKbDXLtlUUKmYtBr73FPo7Mk30NWqtYsDliwDPAS9fIKRYzXYDFIdA5MJiwujlzSsj0/pub?gid=0&single=true&output=csv'
# csv_path=r'../input/astronaut-yearbook/astronauts.csv'
astro_df = pd.read_csv(astro_google_sheets_url)
astro_df = astro_df.fillna('')
astro_df.info()
military_ranks_to_pay_grades = {'ColonelUS Army':'O-6', 'ColonelUS Air Force':'O-6',
       'Lieutenant ColonelUS Marine Corps':'O-5', 'CaptainUS Navy':'O-6',
       'Major GeneralUS Air Force':'O-8', 'Lieutenant ColonelUS Air Force':'O-5',
       'CommanderUS Navy':'O-5', 'CaptainUS Air Force':'O-3',
       'Major GeneralUS Marine Corps':'O-8', 'ColonelUS Marine Corps':'O-6',
       'CaptainUS Coast Guard':'O-6', 'Lieutenant CommanderUS Navy':'O-4', 
       'Brigadier GeneralUS Air Force':'O-7', 'Lieutenant ColonelUS Army':'O-5',
       'MajorUS Air Force':'O-4', 'Lieutenant GeneralUS Air Force':'O-9',
       'Chief Warrant OfficerUS Army':'CWO-4', 'Rear AdmiralUS Navy':'O-8',
       'CommanderUS Coast Guard':'O-5', 'CaptainUS Army':'O-3',
       'Brigadier GeneralUS Army':'O-7', 'Vice AdmiralUS Navy':'O-9',
       'MajorUS Marine Corps':'O-4'}
# Split multivalued fields into lists so will be exported as json arrays
astro_df["Alma Mater"] = astro_df["Alma Mater"].str.split("\s*;\s*")
astro_df["Missions"] = astro_df["Missions"].str.split('\s*,\s*')
astro_df["Undergraduate Major"] = astro_df["Undergraduate Major"].str.split('\s*,\s*')
astro_df["Graduate Major"] = astro_df["Graduate Major"].str.split('\s*,\s*')

# Use nameparser module to split Name field into First, Middle, Last Name.
# namparser handles case that not all rows have middle initial
astro_df[["First Name", "Middle Initial", "Last Name"]] = astro_df["Name"].apply(lambda x: pd.Series({"first":HumanName(x).first, "middle":HumanName(x).middle, "last":HumanName(x).last}))

# Split Birth Place field into City and State/Country
astro_df[["Birth Place City", "Birth Place State/Country"]] = astro_df["Birth Place"].str.split('\s*,\s*', expand=True)

# Consolidate into the primary military services, remove "(Retired)" and "Reserves"
astro_df["Military Branch"] = astro_df["Military Branch"].replace(regex=r'Naval', value='Navy').replace(regex=[r' \(Retired\)$', ' Reserves'], value='')

# Map Military Rank/Branch field into pay grades so we can tell the difference between an Army/USMC/USAF Captain vs a Navy Captain, also helps for sorting
astro_df["Military Pay Grade"] = astro_df["Military Rank"].str.cat(astro_df["Military Branch"]).map(military_ranks_to_pay_grades).fillna('')
astro_json_string = astro_df.to_json(orient="index")
astro_json = json.loads(astro_json_string)
print(json.dumps(astro_json, indent=2))
astronautDataFile = open("astronaut.json", "w")
astronautDataFile.write(json.dumps(astro_json, indent=4))
from IPython.display import FileLink
FileLink(r'./astronaut.json')
