# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from altair import *

import os, io, json



%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
days_map = {0:'Mon',1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

cc_code = {

1: 'United States' ,

7: 'Russia' ,

20: 'Egypt' ,

27: 'South Africa' ,

30: 'Greece' ,

31: 'Netherlands' ,

32: 'Belgium' ,

33: 'France' ,

34: 'Spain' ,

36: 'Hungary' ,

39: 'Italy' ,

40: 'Romania' ,

41: 'Switzerland' ,

43: 'Austria' ,

44: 'United Kingdom' ,

45: 'Denmark' ,

46: 'Sweden' ,

47: 'Norway' ,

48: 'Poland' ,

49: 'Germany' ,

51: 'Peru' ,

52: 'Mexico' ,

53: 'Cuba' ,

54: 'Argentina' ,

55: 'Brazil' ,

56: 'Chile' ,

57: 'Colombia' ,

58: 'Venezuela' ,

60: 'Malaysia' ,

61: 'Australia' ,

62: 'Indonesia' ,

63: 'Philippines' ,

64: 'New Zealand' ,

65: 'Singapore' ,

66: 'Thailand' ,

81: 'Japan' ,

82: 'South Korea' ,

84: 'Vietnam' ,

86: 'China' ,

90: 'Turkey' ,

91: 'India' ,

92: 'Pakistan' ,

93: 'Afghanistan' ,

94: 'Sri Lanka' ,

95: 'Myanmar' ,

98: 'Iran' ,

211: 'South Sudan' ,

212: 'Morocco' ,

213: 'Algeria' ,

216: 'Tunisia' ,

218: 'Libya' ,

220: 'Gambia' ,

221: 'Senegal' ,

222: 'Mauritania' ,

223: 'Mali' ,

224: 'Guinea' ,

225: 'Ivory Coast' ,

226: 'Burkina Faso' ,

227: 'Niger' ,

228: 'Togo' ,

229: 'Benin' ,

230: 'Mauritius' ,

231: 'Liberia' ,

232: 'Sierra Leone' ,

233: 'Ghana' ,

234: 'Nigeria' ,

235: 'Chad' ,

236: 'Central African Republic' ,

237: 'Cameroon' ,

238: 'Cape Verde' ,

239: 'Sao Tome and Principe' ,

240: 'Equatorial Guinea' ,

241: 'Gabon' ,

242: 'Republic of the Congo' ,

243: 'Democratic Republic of the Congo' ,

244: 'Angola' ,

245: 'Guinea-Bissau' ,

246: 'British Indian Ocean Territory' ,

248: 'Seychelles' ,

249: 'Sudan' ,

250: 'Rwanda' ,

251: 'Ethiopia' ,

252: 'Somalia' ,

253: 'Djibouti' ,

254: 'Kenya' ,

255: 'Tanzania' ,

256: 'Uganda' ,

257: 'Burundi' ,

258: 'Mozambique' ,

260: 'Zambia' ,

261: 'Madagascar' ,

262: 'Mayotte' ,

263: 'Zimbabwe' ,

264: 'Namibia' ,

265: 'Malawi' ,

266: 'Lesotho' ,

267: 'Botswana' ,

268: 'Swaziland' ,

269: 'Comoros' ,

290: 'Saint Helena' ,

291: 'Eritrea' ,

297: 'Aruba' ,

298: 'Faroe Islands' ,

299: 'Greenland' ,

333: 'Yemen' ,

334: 'Zambia' ,

335: 'Zimbabwe' ,

350: 'Gibraltar' ,

351: 'Portugal' ,

352: 'Luxembourg' ,

353: 'Ireland' ,

354: 'Iceland' ,

355: 'Albania' ,

356: 'Malta' ,

357: 'Cyprus' ,

358: 'Finland' ,

359: 'Bulgaria' ,

370: 'Lithuania' ,

371: 'Latvia' ,

372: 'Estonia' ,

373: 'Moldova' ,

374: 'Armenia' ,

375: 'Belarus' ,

376: 'Andorra' ,

377: 'Monaco' ,

378: 'San Marino' ,

379: 'Vatican' ,

380: 'Ukraine' ,

381: 'Serbia' ,

382: 'Montenegro' ,

383: 'Kosovo' ,

385: 'Croatia' ,

386: 'Slovenia' ,

387: 'Bosnia and Herzegovina' ,

389: 'Macedonia' ,

420: 'Czech Republic' ,

421: 'Slovakia' ,

423: 'Liechtenstein' ,

500: 'Falkland Islands' ,

501: 'Belize' ,

502: 'Guatemala' ,

503: 'El Salvador' ,

504: 'Honduras' ,

505: 'Nicaragua' ,

506: 'Costa Rica' ,

507: 'Panama' ,

508: 'Saint Pierre and Miquelon' ,

509: 'Haiti' ,

590: 'Saint Barthelemy' ,

591: 'Bolivia' ,

592: 'Guyana' ,

593: 'Ecuador' ,

595: 'Paraguay' ,

597: 'Suriname' ,

598: 'Uruguay' ,

599: 'Netherlands Antilles' ,

670: 'East Timor' ,

672: 'Antarctica' ,

673: 'Brunei' ,

674: 'Nauru' ,

675: 'Papua New Guinea' ,

676: 'Tonga' ,

677: 'Solomon Islands' ,

678: 'Vanuatu' ,

679: 'Fiji' ,

680: 'Palau' ,

681: 'Wallis and Futuna' ,

682: 'Cook Islands' ,

683: 'Niue' ,

685: 'Samoa' ,

686: 'Kiribati' ,

687: 'New Caledonia' ,

688: 'Tuvalu' ,

689: 'French Polynesia' ,

690: 'Tokelau' ,

691: 'Micronesia' ,

692: 'Marshall Islands' ,

850: 'North Korea' ,

852: 'Hong Kong' ,

853: 'Macau' ,

855: 'Cambodia' ,

856: 'Laos' ,

880: 'Bangladesh' ,

886: 'Taiwan' ,

960: 'Maldives' ,

961: 'Lebanon' ,

962: 'Jordan' ,

963: 'Syria' ,

964: 'Iraq' ,

965: 'Kuwait' ,

966: 'Saudi Arabia' ,

967: 'Yemen' ,

968: 'Oman' ,

970: 'Palestine' ,

971: 'United Arab Emirates' ,

972: 'Israel' ,

973: 'Bahrain' ,

974: 'Qatar' ,

975: 'Bhutan' ,

976: 'Mongolia' ,

977: 'Nepal' ,

992: 'Tajikistan' ,

993: 'Turkmenistan' ,

994: 'Azerbaijan' ,

995: 'Georgia' ,

996: 'Kyrgyzstan' ,

998: 'Uzbekistan' ,

1242: 'Bahamas' ,

1246: 'Barbados' ,

1264: 'Anguilla' ,

1268: 'Antigua and Barbuda' ,

1284: 'British Virgin Islands' ,

1340: 'U.S. Virgin Islands' ,

1345: 'Cayman Islands' ,

1441: 'Bermuda' ,

1473: 'Grenada' ,

1649: 'Turks and Caicos Islands' ,

1664: 'Montserrat' ,

1670: 'Northern Mariana Islands' ,

1671: 'Guam' ,

1684: 'American Samoa' ,

1721: 'Sint Maarten' ,

1758: 'Saint Lucia' ,

1767: 'Dominica' ,

1784: 'Saint Vincent and the Grenadines' ,

1809: 'Dominican Republic' ,

1829: 'Dominican Republic' ,

1849: 'Dominican Republic' ,

1868: 'Trinidad and Tobago' ,

1869: 'Saint Kitts and Nevis' ,

1876: 'Jamaica' ,

1939: 'Puerto Rico' ,

441481: 'Guernsey' ,

441534: 'Jersey' ,

441624: 'Isle of Man' 

}
df_cdrs = pd.DataFrame({})

for i in range(1,8):

    df = pd.read_csv('../input/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['datetime'])

    df_cdrs = df_cdrs.append(df)



cell = json.load(io.open('../input/milano-grid.geojson'))

cellid = {}  

for i in range(len(cell['features'])):

    cid = cell['features'][i]['properties']['cellId']

    geom = cell['features'][i]['geometry']

    cellid[cid] = geom  
df_cdrs.fillna(0, inplace=True)

df_cdrs['country_name'] = df_cdrs.countrycode.map(cc_code)

df_cdrs['date'] = df_cdrs.datetime.dt.date

df_cdrs['dayofweek'] = df_cdrs.datetime.dt.dayofweek.map(days_map)

df_cdrs['hour'] = df_cdrs.datetime.dt.hour

df_cdrs['calls'] = df_cdrs['callout'] + df_cdrs['callin']

df_cdrs['sms'] = df_cdrs['smsout'] + df_cdrs['smsin']
df_calls_sms = pd.pivot_table(df_cdrs, index='datetime', aggfunc='sum', fill_value=0, values=['calls', 'sms', 'internet']).unstack().reset_index()

df_calls_sms.columns = ['event_type', 'datetime', 'measure']

Chart(df_calls_sms).mark_area().encode(X('datetime:T', title='Time of Event'), 

                                       Y('measure:Q', title='Number of Events'), 

                                       Row('event_type'),

                                       color='event_type')
df_events_by_country = pd.pivot_table(df_cdrs, index='country_name', columns='hour', values=['calls','sms','internet'], aggfunc=np.sum).unstack().reset_index()

df_events_by_country.columns = ['event_type', 'hour', 'country_name', 'measure']

df_events_by_country.head(20)
def heatmap(data, row, column, color, cellsize=(30, 15)):

    """Create an Altair Heat-Map



    Parameters

    ----------

    row, column, color : str

        Altair trait shorthands

    cellsize : tuple

        specify (width, height) of cells in pixels

    """

    return Chart(data).mark_text(

               applyColorToBackground=True,

           ).encode(

               row=row,

               column=column,

               text=Text(value=' '),

               color=color

           ).configure_scale(

               textBandWidth=cellsize[0],

               bandSize=cellsize[1]

           )
top_20_countries = df_events_by_country.groupby('country_name')['measure'].sum().sort_values().tail(20)

heatmap(

    df_events_by_country[df_events_by_country.country_name.isin(top_20_countries.index)]

    .query('country_name != "Italy"'), row='country_name', column='hour', color='sum(measure)')