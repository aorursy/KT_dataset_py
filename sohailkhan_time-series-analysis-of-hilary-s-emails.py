# The data comes both as CSV files and a SQLite database

import pandas as pd
import numpy as np
persons = pd.read_csv("../input/Persons.csv")
emails  = pd.read_csv("../input/Emails.csv")

hilaryEmails = emails[emails.SenderPersonId == 80]
hilaryEmails = hilaryEmails.dropna(how='any')
date_re = r'(?P<dow>\w+,\s*?)?(?P<month>)'

hilaryEmails.ExtractedDateSent[hilaryEmails.ExtractedDateSent.str.split(" ").apply(len) == 6]
def fixDate(date):
    spl = date.split(" ")
    
    if len(spl) >= 6:
        try:
            if spl[0].startswith(u'\xef\xbf\xbd'):
                spl = spl[1:]
        except UnicodeEncodeError:
            spl = spl[1:]
        dow, month, day, year, time = spl[:5]
    elif len(spl) == 5:
        if spl[-1].endswith('M'):
            return np.NAN
        else:
            dow, month, day, time, year = spl[:5]
    else:
        return np.NAN
    try:
        if ':' not in time:
            time = time[:-2] + ':' + time[-2:]
        return u"{} {} {} {}".format(month, day, year, time.replace('.', ''))
    except UnicodeEncodeError as e:
        print(e)
def tryToCoerce(s):
    try:
        return pd.to_datetime(fixDate(s))
    except Exception:
        return np.NaN
pd.to_datetime(fixDate('Thu Sep 17 06:03:43 2009'))
sum(hilaryEmails.
    ExtractedDateSent
    .apply(tryToCoerce)
    .isnull())
hilaryEmails['cleanedDate'] = (hilaryEmails
                               .ExtractedDateSent.apply(tryToCoerce)
                               .dropna(how="any")
)
hilaryEmails.index = hilaryEmails.cleanedDate
hilaryEmails.sort_index(inplace=True)
minDate, maxDate = hilaryEmails.index.min(), hilaryEmails.index.max()
"Hilary's emails range from {} to {}".format(minDate.date(), maxDate.date())
hilaryEmails.resample('D', how='count').Id.plot()
# before 2011 and after counts
hilaryEmails[:'2011-01'].Id.count(), hilaryEmails['2011-01':].Id.count()
hilaryEmails.Id.resample("D", how="count").sort_values(ascending=False).head(1)

hilaryEmails['2009-08-29'][['MetadataSubject', 
                            'ExtractedBodyText']]
hilaryEmails.groupby(lambda s: s.hour).apply(len)