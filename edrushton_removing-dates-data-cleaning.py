import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
pd.set_option('display.max_rows', 40)
nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 
nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero
nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 
                            # that people doon't generally use all number format for old dates, but write them out 
nDELIM = r'(?:[\/\-\._])?'  # 
NUM_DATE = f"""
    (?:
        # YYYY-MM-DD
        (?:{nYR}{nDELIM}{nMNTH}{nDELIM}{nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}{nDELIM}{nDAY}{nDELIM}{nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}{nDELIM}{nMNTH}{nDELIM}{nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}{nDELIM}{nDAY}{nDELIM}{nYR})
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE | re.UNICODE)
test_dates = ['15:12', '15:30, 12th March 1972 and then 3rd May 2014.', 'May', 'April 2006',
 't 6', 'a few words before 31/12/2019 ',
 '99.10.10.8', '2016.31.55',
 'on 2012-09-11 and a few words after ',
 '1234-56-78', '2 t', '04:27, 20 Jan 2004',
 '10, 9 March 2011', '20:04, 17 Dec 2004',
 'wed at', 't 2005', '2012/56/10', '5th', 't 1600', 'on August, 2014',
 'May 25, 1992', 'on mar', '1, 4 , 5', '03:19, 1 December',
 'on 2 October', '1930', '15:59, 17 December',
 '5-5-5', 'the time is 12:12:54 am ', '17:80pm', '10:52, 11',
 '10 second', '6 of t', '17:09, 16 Jun 2005',
 '19:32', '20th', 'Dec of 04', '1978 T',
 '09/11', 't133', 't 21:31', 't on 29 October 2013 T',
 '19:53, 15', 'wed by t', '3/4 of t', '149,000', '29 August 2006', 't Mar',
 'November 2014', '7 of t', 'December 11 2006', '14:59, 16',
 'mon to', '02:08, Mar 13, 2004', '69 of', 'of nov',
 '/may-2012', '3am', 'of March', '2003',
 'on 12 Dec', 't 2', 'at mar', 't of 1993',
 '2790', 't, may', '21:51, January 11, 2016',
 '22 May 2005', '16 December 2005', 'July, 1870', 'On Dec 14, 2006',
 't 3 of t', '07:51, 2004 Jul 16', 't dec', 'April 2006 T',
 't, 2012', 'March 16, 1869', 'wed to', '20 January 2013',
 '26t', 't-may','4000']
pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])
NUM_DATE = f"""
    (?P<num_date>
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE)
pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])
NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE)
pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])
DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number, but without a leading zero
    (?:[123]?\d)
)"""
MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""
DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'
DATE_PATTERN = f"""(?P<wordy_date>
    (?:^|\W)
    (?:
    (?:
        # Year - Day - Month (or just Day - Month)
        (?:{YEAR}{DELIM})?(?:{DAY}{DELIM}{MONTH})
        |
        # Month - Day - Year (or just Month - Day)
        (?:{MONTH}{DELIM}{DAY})(?:{DELIM}{YEAR})?
    )
    | 
        # Just Month - Year
    (?:{MONTH}{DELIM}{YEAR})
    )
    (?:$|\W)
)"""

# Let's combine with the numbers only version from above
myDate = re.compile(f'{DATE_PATTERN}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': '@@'.join(myDate.findall(txt))} for txt in test_dates])
YEAR_4D = r"""(?:[12]\d\d\d)"""
DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

myDate = re.compile(f'{DATE_PATTERN}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])
TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon
:[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""
myDate = re.compile(f'{TIME}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': '@@'.join(myDate.findall(txt))} for txt in test_dates])
COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)
pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])
def findCombined(txt):
    myResults = []
    for matchGroup in myDate.finditer(txt):
        myResults.append(matchGroup.group('combined'))
    return myResults

pd.DataFrame([{'test_text': txt, 'match': findCombined(txt)} for txt in test_dates])
pd.DataFrame([{'test_text': txt, 'match': findCombined(txt), 'subbed': myDate.sub(' xxDATExx ', txt)} for txt in test_dates])
# Let's use tqdm to givec us a nice progress bar
from tqdm import tqdm
tqdm.pandas(tqdm)
for df in [train, test]:
    df['fewer_dates'] = df.comment_text.progress_apply(lambda x: myDate.sub(' xxDATExx ', x))
# First change the display options so we can see the entire comments
pd.set_option('display.max_colwidth', -1)
train.loc[train.fewer_dates.str.contains('xxDATExx'), ['comment_text', 'fewer_dates']].head()

print('Found {} rows with dates in the training set'.format(train.fewer_dates.str.contains('xxDATExx').sum()))
print('Found {} rows with dates in the test set'.format(test.fewer_dates.str.contains('xxDATExx').sum()))
      
nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 
nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero
nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 
                            # that people doon't generally use all number format for old dates, but write them out 
nDELIM = r'(?:[\/\-\._])?'  # 
NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""
DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number, but without a leading zero
    (?:[123]?\d)
)"""
MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""
DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""
DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon
:[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)
