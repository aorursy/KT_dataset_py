# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pycountry #country codes conversion



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if "import" in filename:

            df_import = pd.read_csv(os.path.join(dirname, filename))

        else:

            df_export = pd.read_csv(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_import.describe()
df_export.describe()
df_import.drop_duplicates(keep="first", inplace=True)
df_import.dropna(subset=['value'], inplace=True)
df_export.dropna(subset=['value'], inplace=True)
df_export.head()
countrydict = {'AFGHANISTAN TIS': 'af', 'ALBANIA': 'al', 'ALGERIA': 'dz', 'AMERI SAMOA': 'us', 'ANDORRA': 'ad', 'ANGOLA': 'ao', 'ANGUILLA': 'ai', 'ANTARTICA': 'aq', 'ANTIGUA': 'ag', 'ARGENTINA': 'ar', 'ARMENIA': 'am', 'ARUBA': 'aw', 'AUSTRALIA': 'au', 'AUSTRIA': 'at', 'AZERBAIJAN': 'az', 'BAHAMAS': 'bs', 'BAHARAIN IS': 'bh', 'BANGLADESH PR': 'bd', 'BARBADOS': 'bb', 'BELARUS': 'by', 'BELGIUM': 'be', 'BELIZE': 'bz', 'BENIN': 'bj', 'BERMUDA': 'bm', 'BHUTAN': 'bt', 'BOLIVIA': 'bo', 'BOSNIA-HRZGOVIN': 'ba', 'BOTSWANA': 'bw', 'BR VIRGN IS': 'vg', 'BRAZIL': 'br', 'BRUNEI': 'bn', 'BULGARIA': 'bg', 'BURKINA FASO': 'bf', 'BURUNDI': 'bi', 'C AFRI REP': 'cf', 'CAMBODIA': 'kh', 'CAMEROON': 'cm', 'CANADA': 'ca', 'CAPE VERDE IS': 'cv', 'CAYMAN IS': 'ky', 'CHAD': 'td', 'CHILE': 'cl', 'CHINA P RP': 'cn', 'CHRISTMAS IS.': 'cx', 'COCOS IS': 'cc', 'COLOMBIA': 'co', 'COMOROS': 'km', 'CONGO D. REP.': 'cd', 'CONGO P REP': 'cg', 'COOK IS': 'ck', 'COSTA RICA': 'cr', 'COTE D\' IVOIRE': 'ci', 'CROATIA': 'hr', 'CUBA': 'cu', 'CYPRUS': 'cy', 'CZECH REPUBLIC': 'cz', 'DENMARK': 'dk', 'DJIBOUTI': 'dj', 'DOMINIC REP': 'do', 'DOMINICA': 'do', 'ECUADOR': 'ec', 'EGYPT A RP': 'eg', 'EL SALVADOR': 'sv', 'EQUTL GUINEA': 'gq', 'ERITREA': 'er', 'ESTONIA': 'ee', 'ETHIOPIA': 'et', 'FALKLAND IS': 'fk', 'FAROE IS.': 'fo', 'FIJI IS': 'fj', 'FINLAND': 'fi', 'FR GUIANA': 'gf', 'FR POLYNESIA': 'fr', 'FR S ANT TR': 'fr', 'FRANCE': 'fr', 'GABON': 'ga', 'GAMBIA': 'gm', 'GEORGIA': 'ge', 'GERMANY': 'de', 'GHANA': 'gh', 'GIBRALTAR': 'gi', 'GREECE': 'gr', 'GREENLAND': 'gl', 'GRENADA': 'gd', 'GUADELOUPE': 'gp', 'GUAM': 'gu', 'GUATEMALA': 'gt', 'GUERNSEY': 'gg', 'GUINEA BISSAU': 'gw', 'GUINEA': 'gn', 'GUYANA': 'gy', 'HAITI': 'ht', 'HEARD MACDONALD': 'hm', 'HONDURAS': 'hn', 'HONG KONG': 'hk', 'HUNGARY': 'hu', 'ICELAND': 'is', 'INDONESIA': 'id', 'IRAN': 'ir', 'IRAQ': 'iq', 'IRELAND': 'ie', 'ISRAEL': 'il', 'ITALY': 'it', 'JAMAICA': 'jm', 'JAPAN': 'jp', 'JERSEY         ': 'je', 'JORDAN': 'jo', 'KAZAKHSTAN': 'kz', 'KENYA': 'ke', 'KIRIBATI REP': 'ki', 'KOREA DP RP': 'kp', 'KOREA RP': 'kr', 'KUWAIT': 'kw', 'KYRGHYZSTAN': 'kg', 'LAO PD RP': 'la', 'LATVIA': 'lv', 'LEBANON': 'lb', 'LESOTHO': 'ls', 'LIBERIA': 'lr', 'LIBYA': 'ly', 'LIECHTENSTEIN': 'li', 'LITHUANIA': 'lt', 'LUXEMBOURG': 'lu', 'MACAO': 'mo', 'MACEDONIA': 'mk', 'MADAGASCAR': 'mg', 'MALAWI': 'mw', 'MALAYSIA': 'my', 'MALDIVES': 'mv', 'MALI': 'ml', 'MALTA': 'mt', 'MARSHALL ISLAND': 'mh', 'MARTINIQUE': 'mq', 'MAURITANIA': 'mr', 'MAURITIUS': 'mu', 'MAYOTTE': 'yt', 'MEXICO': 'mx', 'MICRONESIA': 'fm', 'MOLDOVA': 'md', 'MONACO': 'mc', 'MONGOLIA': 'mn', 'MONTENEGRO': 'me', 'MONTSERRAT': 'ms', 'MOROCCO': 'ma', 'MOZAMBIQUE': 'mz', 'MYANMAR': 'mm', 'N. MARIANA IS.': 'mp', 'NAMIBIA': 'na', 'NAURU RP': 'nr', 'NEPAL': 'np', 'NETHERLAND': 'nl', 'NETHERLANDANTIL': 'nl', 'NEW CALEDONIA': 'cn', 'NEW ZEALAND': 'nz', 'NICARAGUA': 'ni', 'NIGER': 'ne', 'NIGERIA': 'ng', 'NIUE IS': 'nu', 'NORFOLK IS': 'nf', 'NORWAY': 'no', 'OMAN': 'om', 'PACIFIC IS]': 'ot', 'PAKISTAN IR': 'pk', 'PALAU': 'pw', 'PANAMA C Z': 'pa', 'PANAMA REPUBLIC': 'pa', 'PAPUA N GNA': 'pg', 'PARAGUAY': 'py', 'PERU': 'pe', 'PHILIPPINES': 'ph', 'PITCAIRN IS.': 'pn', 'POLAND': 'pi', 'PORTUGAL': 'pt', 'PUERTO RICO': 'pr', 'QATAR': 'qa', 'REUNION': 're', 'ROMANIA': 'ro', 'RUSSIA': 'ru', 'RWANDA': 'rw', 'SAMOA': 'ws', 'SAN MARINO': 'sm', 'SAO TOME': 'st', 'SAUDI ARAB': 'sa', 'SENEGAL': 'sn', 'SERBIA': 'rs', 'SEYCHELLES': 'sc', 'SIERRA LEONE': 'si', 'SINGAPORE': 'sg', 'SLOVAK REP': 'sk', 'SLOVENIA': 'si', 'SOLOMON IS': 'sb', 'SOMALIA': 'so', 'SOUTH AFRICA': 'za', 'SOUTH SUDAN ': 'sd', 'SPAIN': 'es', 'SRI LANKA DSR': 'lk', 'ST HELENA': 'sh', 'ST KITT N A': 'kn', 'ST LUCIA': 'lc', 'ST PIERRE': 'pm', 'ST VINCENT': 'vc', 'STATE OF PALEST': 'ps', 'SUDAN': 'sd', 'SURINAME': 'sr', 'SWAZILAND': 'sz', 'SWEDEN': 'se', 'SWITZERLAND': 'ch', 'SYRIA': 'sy', 'TAIWAN': 'tw', 'TAJIKISTAN': 'tj', 'TANZANIA REP': 'tz', 'THAILAND': 'th', 'TIMOR LESTE': 'tl', 'TOGO': 'tg', 'TOKELAU IS': 'tk', 'TONGA': 'to', 'TRINIDAD': 'tt', 'TUNISIA': 'tn', 'TURKEY': 'tr', 'TURKMENISTAN': 'tm', 'TURKS C IS': 'tr', 'TUVALU': 'tv', 'U ARAB EMTS': 'ae', 'U K': 'gb', 'U S A': 'us', 'UGANDA': 'ug', 'UKRAINE': 'ua', 'UNION OF SERBIA & MONTENEGRO': 'rs', 'UNSPECIFIED': 'ot', 'URUGUAY': 'uy', 'US MINOR OUTLYING ISLANDS               ': 'us', 'UZBEKISTAN': 'uz', 'VANUATU REP': 'vu', 'VATICAN CITY': 'va', 'VENEZUELA': 've', 'VIETNAM SOC REP': 'vn', 'VIRGIN IS US': 'us', 'WALLIS F IS': 'wf', 'YEMEN REPUBLC': 'ye', 'ZAMBIA': 'zm', 'ZIMBABWE': 'zw', 'SAHARWI A.DM RP': 'ot', 'NEUTRAL ZONE': 'ot', 'CANARY IS': 'ic', 'PACIFIC IS': 'ot', 'CHANNEL IS': 'je', 'INSTALLATIONS IN INTERNATIONAL WATERS   ': 'ot', 'SINT MAARTEN (DUTCH PART)': 'nl', 'CURACAO': 'cw'}
df_import.replace(to_replace=countrydict,inplace=True)

df_export.replace(to_replace=countrydict,inplace=True)
binend=(0,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,96,100)

binlabels=('Animals & Animal Products',

'Vegetable Products',

'Animal Or Vegetable Fats',

'Prepared Foodstuffs',

'Mineral Products',

'Chemical Products',

'Plastics & Rubber',

'Hides & Skins',

'Wood & Wood Products',

'Wood Pulp Products',

'Textiles & Textile Articles',

'Footwear, Headgear',

'Articles Of Stone, Plaster, Cement, Asbestos',

'Pearls, Precious Or Semi-Precious Stones, Metals',

'Base Metals & Articles Thereof',

'Machinery & Mechanical Appliances',

'Transportation Equipment',

'Instruments - Measuring, Musical',

'Arms & Ammunition',

'Miscellaneous',

'Works Of Art')

df_import['cat'] = pd.cut(df_import['HSCode'],binend, False, binlabels)

df_export['cat'] = pd.cut(df_export['HSCode'],binend, False, binlabels)
countryexports = df_export.groupby('country').agg({'value':'sum'})

countryimports = df_import.groupby('country').agg({'value':'sum'})
countryexports.rename({"value":"export"}, axis=1, inplace=True)

countryimports.rename({"value":"import"}, axis=1, inplace=True)
df_alldata = pd.merge(countryimports,countryexports, on='country')

df_alldata['td'] = df_alldata['export'] - df_alldata['import']

df_alldata = df_alldata.reset_index()
def getcountry(alpha2,nametype):

    try:

        return getattr(pycountry.countries.get(alpha_2=alpha2.upper()),nametype)

    except:

        np.NaN

df_alldata['fullcountry'] = df_alldata['country'].apply(lambda x: getcountry(x, "name"))

df_alldata['country'] = df_alldata['country'].apply(lambda x: getcountry(x,"alpha_3"))
import plotly.express as px

import plotly as py

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)
df_alldata[df_alldata['country']=="CAN"]
import plotly.graph_objects as go



df = df_alldata



fig = go.Figure(data=go.Choropleth(

    locations = df['country'],

    z = df['export'],

    text = df['fullcountry'],

    colorscale = 'darkmint',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '$',

    colorbar_title = 'Exports<br>Millions US$',

))



fig.update_layout(

    title_text='Exports to India 2010-2018',

    geo=dict(

        showframe=True,

        showcoastlines=True,

        projection_type='equirectangular'

    )

)
import plotly.graph_objects as go



df = df_alldata



fig = go.Figure(data=go.Choropleth(

    locations = df['country'],

    z = df['import'],

    text = df['fullcountry'],

    colorscale = 'magenta',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '$',

    colorbar_title = 'Imports<br>Millions US$',

))



fig.update_layout(

    title_text='Imports to India 2010-2018',

    geo=dict(

        showframe=True,

        showcoastlines=True,

        projection_type='equirectangular'

    )

)
import plotly.graph_objects as go



df = df_alldata



fig = go.Figure(data=go.Choropleth(

    locations = df['country'],

    z = df['td'],

    text = df['fullcountry'],

    colorscale = 'thermal',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_tickprefix = '$',

    colorbar_title = 'Trade Deficit 2010-2018 Cumulative<br>Millions US$',

))



fig.update_layout(

    title_text='Trade Deficit with India 2010-2018',

    geo=dict(

        showframe=True,

        showcoastlines=True,

        projection_type='equirectangular'

    )

)