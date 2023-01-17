import pandas as pd
# Import semi clean walmart data
clean_df = pd.read_csv('../input/walmart_semiclean.csv') 
# Check the data types for the imported data frame
clean_df.info()
# Lower case all column values
# Rename IsHoliday column to 'holiday': Refer to the documentation on renaming columns: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
clean_df.columns = [col.lower() for col in clean_df.columns]
clean_df.rename(columns=dict(isholiday='holiday'),inplace=True)
# Convert Yes and No values to True/False Boolean values
def toBools(val):
    if val == 'Yes':
        return True
    else:
        return False

clean_df.holiday = clean_df.holiday.apply(toBools)
# Convert Dept to integer
clean_df.dept = clean_df.dept.astype(int)
# Convert weekly_sales to float:
# Hint: Before converting to float, make sure there are only numbers that are in the string!

def toFloat(val):
    val = val.replace("'","")
    return float(val)
    
clean_df.weekly_sales = clean_df.weekly_sales.apply(toFloat)
# Convert date to actual datetime.datetime objects
# Hint: Use the helper function in Walkthrough 4-2 that was provided to you

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
# Make sure all values in date column conform to what the helper function expects
pd.options.display.max_rows = 500
clean_df.date.value_counts()
# Label the rows with date value of 0 to be "Invalid Date"
clean_df.date = clean_df.date.where(clean_df.date != '0','Invalid Date')
# Save the Invalid Date rows to a separate dataframe for future analysis. Make to sure to copy the DataFrame to prevent future reference issues
invalids_df = clean_df[clean_df.date == 'Invalid Date'].copy()

# Get rid of the Invalid Date rows in the clean_df DataFrame. Make to sure to copy the DataFrame to prevent future reference issues
clean_df = clean_df[clean_df.date != 'Invalid Date'].copy()
# Convert the dates - This may take around ~10 seconds
clean_df.date = clean_df.date.apply(toDate)
# Confirm that all data has been cleaned and converted to the proper date types
display(clean_df.info()) # display function allows you to display multiple pretty outputs in one cell
display(clean_df.head())