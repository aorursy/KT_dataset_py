import glob

import pandas

import pandasql



def find_file():

    return glob.glob("../input/**/*.xlsx", recursive=True)[0]



def run_sql(query):

    return pandasql.sqldf(query, globals())



"----MONTHS AND ZODIACS----"



Months = pandas.read_excel(find_file(), sheet_name="Months")

print(Months)



Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")

print(Zodiacs)



Months = run_sql("""

    select *

    from Months, Zodiacs

    where Months.ZodiacID='Zodiacs.ZodiacID'

""")



"----MONTHS AND GEMSTONES----"



Months = pandas.read_excel(find_file(), sheet_name="Months")

print(Months)



Gemstones = pandas.read_excel(find_file(), sheet_name="Gemstones")

print(Gemstones)



Months = run_sql("""

    select *

    from Months, Gemstones

    where Months.GemstoneID='Gemstones.GemstoneID'

""")



"----GEMSTONES AND ZODIACS----"



Gemstones = pandas.read_excel(find_file(), sheet_name="Gemstones")

print(Gemstones)



Zodiacs = pandas.read_excel(find_file(), sheet_name="Zodiacs")

print(Zodiacs)



Gemstones = run_sql("""

    select *

    from Gemstones, Zodiacs

    where Gemstones.GemstoneID='Zodiacs.ZodiacID'

""")