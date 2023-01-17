import re
f = open("../input/evendates/evendates.txt", "r")

data = (f.readlines())

f.close()

dates = "".join(data).split('\n')

print(dates[:5])

print(len(dates))
patterns = []

patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}$')

patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}$')

patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}\s{0,2}\d{2}(:|.)\d{2}(:|.)\d{2}(:|.)\d{2,3}$')

patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}\s{0,2}\d{2}(:|.)\d{2}(:|.)\d{2}$')

patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}T\d{1,2}(:|.)\d{1,2}(:|.)\d{1,2}$')



patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}\s{0,2}\d{1,2}(:|.)\d{2}\s{0,2}(AM|PM)\s{0,2}EST$')

patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}\s{0,2}\d{1,2}(:|.)\d{2}\s{0,2}(AM|PM)\s{0,2}EDT$')



patterns.append(r'^[A-Z]{3}\s{0,2}\d{1,2},?\s{0,2}\d{4}$')

patterns.append(r'^[A-Z]{3}\s{0,2}\d{1,2},?\s{0,2}\d{4}\s{0,2}\d{1,2}(:|.)\d{1,2}\s{0,2}(AM|PM)$')





match = []

non_match = []

        

for date in dates:

    matched = False

    for pattern in patterns:

        compiler = re.compile(pattern,flags=re.IGNORECASE)

        if compiler.search(date):

            match.append((date,pattern))

            matched = True

    if not matched:

        non_match.append(date)

            



# Let;s check the length of matched dates and non matched dates

print(len(match), len(non_match))

# print invalid dates

print(non_match)
# print the matched dates alongwith their format pattern they matched with

print(match)
import datetime 

patterns = []





patterns.append(r'(?P<year>\d{4})(-|/)(?P<month>([0][1-9]|[^0][0-2]|[1-9]))(-|/)(?P<day>\d{1,2})')

patterns.append(r'(?P<year>\d{4})(-|/)(?P<month>([0][1-9]|[^0][0-2]|[1-9]))(-|/)(?P<day>\d{1,2})\s{0,2}(?P<hr>\d{2})(:|.)(?P<min>\d{2})(:|.)(?P<sec>\d{2})(:|.)(?P<ms>\d{2,3})')

dateParts = []



for date in dates:

    for pattern in patterns:

        compiler = re.compile(pattern,flags=re.IGNORECASE)

        if compiler.search(date):

            result = compiler.search(date)

            print(date)

            if result.group("year"):

                year = (int)(result.group("year"))  

            if result.group("day"):

                month = (int)(result.group("month"))

            if result.group("month"):

                day = (int)(result.group("day"))

            #year =          

            #month = (int)(result.group("month"))

            #day = (int)(result.group("day"))

            #date = datetime.datetime(year,month,day)

            dateParts.append(str(datetime.datetime(year,month,day)))

            print(result.groupdict())

            

dateParts