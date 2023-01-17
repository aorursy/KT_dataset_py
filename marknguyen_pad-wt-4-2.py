import pandas as pd
# Import semiclean data from previous walkthrough
clean_df = pd.read_csv('../input/transactions_semiclean.csv')
clean_df.head(5)
## Label date values that missing to be "Missing Date"

clean_df.head(5)
## Save the Missing Date rows to another DataFrame for later analysis

missing_df.head(5)
## Get rid of the Missing Date values in clean_df

clean_df.head(5)
## Look at all possible string formats for the date column
pd.options.display.max_rows = 150



# Notes of possible date formats
# M12.D24.Y2010
# 12-17-2010T00:00:00
# 2012.29.06
# Create function to convert all possible date formats seen to an actual Date object in Python. Date objects are useful to do time-based analysis because
# Date objects can be added and subtracted easily similar to Excel date manipulations.
# You should go over this code on your own time if you're curious of how this function works

import re # Import regex package: Read more about regex expressions here: https://en.wikipedia.org/wiki/Regular_expression
import dateutil #Import a utility to convert date strings to 

def toDate(date):
    """Convert a given date_str to a python Date object
    
    Args:
        date (String): String value in the 3 possible date formats:
                          M12.D24.Y2010
                          12-17-2010T00:00:00
                          2012.29.06
    Return:
        datetime.datetime: Date object from the given string
    """
    
    # Use regular expressions to match on date format: 2010.14.02
    # https://docs.python.org/3.4/library/re.html#regular-expression-syntax
    regex = r"\d{4}.\d{2}.\d{2}"
    match = re.match(regex,date)
    
    if (match):
        # Year and date comes before month
        pi = dateutil.parser.parserinfo(dayfirst=True,yearfirst=True)
        return dateutil.parser.parse(date,pi)
    
    # Use regular expressions to match on date format: M12.D12.Y2014
    regex = r"M\d{2}.D\d{2}.Y\d{4}"
    match = re.match(regex,date)
    
    if (match):
        month,day,year = date.split('.')
        month = month[1:]
        day = day[1:]
        year = year[1:]
        return dateutil.parser.parse(month + "." + day + "." + year)
    
    # Else format must be of the form: 12-17-2010T00:00:00
    return dateutil.parser.parse(date)
## Convert dates to actual date objects

## Confirm that the date column contains datetime objects

clean_df.head(5)