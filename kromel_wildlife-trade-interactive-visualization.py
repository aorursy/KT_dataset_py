## Libraries
# Data manipulation
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1) # it will not cut text in html

# Data visualization
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

from IPython.display import HTML

# Read data
species = pd.read_csv('../input/comptab_2018-01-29 16_00_comma_separated.csv')

# Country codes
# I was not able to download the file here, so I just insert a copied dictionary

country_codes_dict = dict({'AD': 'AND',
 'AE': 'ARE',
 'AF': 'AFG',
 'AG': 'ATG',
 'AI': 'AIA',
 'AL': 'ALB',
 'AM': 'ARM',
 'AN': 'ANT',
 'AO': 'AGO',
 'AQ': 'ATA',
 'AR': 'ARG',
 'AS': 'ASM',
 'AT': 'AUT',
 'AU': 'AUS',
 'AW': 'ABW',
 'AZ': 'AZE',
 'BA': 'BIH',
 'BB': 'BRB',
 'BD': 'BGD',
 'BE': 'BEL',
 'BF': 'BFA',
 'BG': 'BGR',
 'BH': 'BHR',
 'BI': 'BDI',
 'BJ': 'BEN',
 'BM': 'BMU',
 'BN': 'BRN',
 'BO': 'BOL',
 'BR': 'BRA',
 'BS': 'BHS',
 'BT': 'BTN',
 'BV': 'BVT',
 'BW': 'BWA',
 'BY': 'BLR',
 'BZ': 'BLZ',
 'CA': 'CAN',
 'CC': 'CCK',
 'CD': 'COD',
 'CF': 'CAF',
 'CG': 'COG',
 'CH': 'CHE',
 'CI': 'CIV',
 'CK': 'COK',
 'CL': 'CHL',
 'CM': 'CMR',
 'CN': 'CHN',
 'CO': 'COL',
 'CR': 'CRI',
 'CU': 'CUB',
 'CV': 'CPV',
 'CX': 'CXR',
 'CY': 'CYP',
 'CZ': 'CZE',
 'DE': 'DEU',
 'DJ': 'DJI',
 'DK': 'DNK',
 'DM': 'DMA',
 'DO': 'DOM',
 'DZ': 'DZA',
 'EC': 'ECU',
 'EE': 'EST',
 'EG': 'EGY',
 'EH': 'ESH',
 'ER': 'ERI',
 'ES': 'ESP',
 'ET': 'ETH',
 'FI': 'FIN',
 'FJ': 'FJI',
 'FK': 'FLK',
 'FM': 'FSM',
 'FO': 'FRO',
 'FR': 'FRA',
 'GA': 'GAB',
 'GB': 'GBR',
 'GD': 'GRD',
 'GE': 'GEO',
 'GF': 'GUF',
 'GG': 'GGY',
 'GH': 'GHA',
 'GI': 'GIB',
 'GL': 'GRL',
 'GM': 'GMB',
 'GN': 'GIN',
 'GP': 'GLP',
 'GQ': 'GNQ',
 'GR': 'GRC',
 'GS': 'SGS',
 'GT': 'GTM',
 'GU': 'GUM',
 'GW': 'GNB',
 'GY': 'GUY',
 'HK': 'HKG',
 'HM': 'HMD',
 'HN': 'HND',
 'HR': 'HRV',
 'HT': 'HTI',
 'HU': 'HUN',
 'ID': 'IDN',
 'IE': 'IRL',
 'IL': 'ISR',
 'IM': 'IMN',
 'IN': 'IND',
 'IO': 'IOT',
 'IQ': 'IRQ',
 'IR': 'IRN',
 'IS': 'ISL',
 'IT': 'ITA',
 'JE': 'JEY',
 'JM': 'JAM',
 'JO': 'JOR',
 'JP': 'JPN',
 'KE': 'KEN',
 'KG': 'KGZ',
 'KH': 'KHM',
 'KI': 'KIR',
 'KM': 'COM',
 'KN': 'KNA',
 'KP': 'PRK',
 'KR': 'KOR',
 'KW': 'KWT',
 'KY': 'CYM',
 'KZ': 'KAZ',
 'LA': 'LAO',
 'LB': 'LBN',
 'LC': 'LCA',
 'LI': 'LIE',
 'LK': 'LKA',
 'LR': 'LBR',
 'LS': 'LSO',
 'LT': 'LTU',
 'LU': 'LUX',
 'LV': 'LVA',
 'LY': 'LBY',
 'MA': 'MAR',
 'MC': 'MCO',
 'MD': 'MDA',
 'ME': 'MNE',
 'MG': 'MDG',
 'MH': 'MHL',
 'MK': 'MKD',
 'ML': 'MLI',
 'MM': 'MMR',
 'MN': 'MNG',
 'MO': 'MAC',
 'MP': 'MNP',
 'MQ': 'MTQ',
 'MR': 'MRT',
 'MS': 'MSR',
 'MT': 'MLT',
 'MU': 'MUS',
 'MV': 'MDV',
 'MW': 'MWI',
 'MX': 'MEX',
 'MY': 'MYS',
 'MZ': 'MOZ',
 'NA': 'NAM',
 'NC': 'NCL',
 'NE': 'NER',
 'NF': 'NFK',
 'NG': 'NGA',
 'NI': 'NIC',
 'NL': 'NLD',
 'NO': 'NOR',
 'NP': 'NPL',
 'NR': 'NRU',
 'NU': 'NIU',
 'NZ': 'NZL',
 'OM': 'OMN',
 'PA': 'PAN',
 'PE': 'PER',
 'PF': 'PYF',
 'PG': 'PNG',
 'PH': 'PHL',
 'PK': 'PAK',
 'PL': 'POL',
 'PM': 'SPM',
 'PN': 'PCN',
 'PR': 'PRI',
 'PS': 'PSE',
 'PT': 'PRT',
 'PW': 'PLW',
 'PY': 'PRY',
 'QA': 'QAT',
 'RE': 'REU',
 'RO': 'ROU',
 'RS': 'SRB',
 'RU': 'RUS',
 'RW': 'RWA',
 'SA': 'SAU',
 'SB': 'SLB',
 'SC': 'SYC',
 'SD': 'SDN',
 'SE': 'SWE',
 'SG': 'SGP',
 'SH': 'SHN',
 'SI': 'SVN',
 'SJ': 'SJM',
 'SK': 'SVK',
 'SL': 'SLE',
 'SM': 'SMR',
 'SN': 'SEN',
 'SO': 'SOM',
 'SR': 'SUR',
 'ST': 'STP',
 'SV': 'SLV',
 'SY': 'SYR',
 'SZ': 'SWZ',
 'TC': 'TCA',
 'TD': 'TCD',
 'TF': 'ATF',
 'TG': 'TGO',
 'TH': 'THA',
 'TJ': 'TJK',
 'TK': 'TKL',
 'TL': 'TLS',
 'TM': 'TKM',
 'TN': 'TUN',
 'TO': 'TON',
 'TR': 'TUR',
 'TT': 'TTO',
 'TV': 'TUV',
 'TW': 'TWN',
 'TZ': 'TZA',
 'UA': 'UKR',
 'UG': 'UGA',
 'UM': 'UMI',
 'US': 'USA',
 'UY': 'URY',
 'UZ': 'UZB',
 'VA': 'VAT',
 'VC': 'VCT',
 'VE': 'VEN',
 'VG': 'VGB',
 'VI': 'VIR',
 'VN': 'VNM',
 'VU': 'VUT',
 'WF': 'WLF',
 'WS': 'WSM',
 'YE': 'YEM',
 'YT': 'MYT',
 'ZA': 'ZAF',
 'ZM': 'ZMB',
 'ZW': 'ZWE'})
appendix = species['App.'].value_counts().sort_index()

data = [
    go.Bar(x=appendix.index, y=appendix,
           textfont=dict(size=16, color='#333'),
           marker=dict(
               line=dict(
                    color='rgba(50,25,25,0.5)',
                    width=1.5)
           ))
]

layout = go.Layout(
    autosize=False,
    width=600,
    height=400,
    title='The most traded by apendix'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='apendix')
classes = species['Class'].value_counts()

data = [
    go.Bar(x=classes.index, y=classes,
           textfont=dict(size=16, color='#333'),
           marker=dict(
               line=dict(width=1.5)
           ))
]

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='The most traded by class'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='classes')
terms = species['Term'].value_counts()[:10]

data = [
    go.Bar(x=terms.index, y=terms,
           textfont=dict(size=16, color='#333'),
           marker=dict(
               line=dict(
                    width=1.5)
           ))
]

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='The most traded by term'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='terms')
importers = species['Importer'].value_counts()
exporters = species['Exporter'].value_counts()
total_trade = exporters.add(importers, fill_value=0)
total_trade = total_trade.rename(country_codes_dict)

data = [ dict(
        type = 'choropleth',
        locations = total_trade.index,
        z = total_trade,
        colorscale='Electric',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Number of entries'),
      ) ]

layout = dict(
    title = 'Countries involved into wildlife trade<br>(export + import entries)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='trade-world-map' )
purpose_dict = dict({'B': 'Breeding in captivity or artificial propagation',
                    'E': 'Educational',
                    'G': 'Botanical garden',
                    'H': 'Hunting trophy',
                    'L': 'Law enforcement',
                    'M': 'Medical',
                    'N': 'Reintroduction or introduction into the wild',
                    'P': 'Personal',
                    'Q': 'Circus or travelling exhibition',
                    'S': 'Scientific',
                    'T': 'Commercial',
                    'Z': 'Zoo'})
purpose = species['Purpose'].value_counts()

labels = [purpose_dict[purpose] for purpose in purpose.index]
data = [
    go.Bar(x=labels, y=purpose,
           textfont=dict(size=16, color='#333'),
           marker=dict(
               line=dict(
                    width=1.5)
           ))
]

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='The most traded by purpose'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='traded_by_purpose')
source_dict = dict({'A': 'Artificially propagated',
                    'C': 'Bred in captivity',
                    'D': 'Bred in captivity for commercial purposes',
                    'F': 'Born in captivity',
                    'I': 'Confiscated or seized specimens',
                    'O': 'Pre-Convention specimens',
                    'R': 'Ranched specimens',
                    'U': 'Unknown',
                    'W': 'Taken from wild',
                    'X': 'Taken from marine env.'})
source = species['Source'].value_counts()

labels = [source_dict[source] for source in source.index]
data = [
    go.Bar(x=labels, y=purpose,
           textfont=dict(size=16, color='#333'),
           marker=dict(
               line=dict(
                    width=1.5)
           ))
]

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='The most traded by source'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='traded_by_source')
exporters_10 = species['Exporter'].value_counts()[:10]
classes_all = species['Class'].value_counts()

missing_values = []
all_values = 0
for i in range(len(exporters_10)):
    missing_values.append(species[species['Exporter'] == exporters_10.index[i]]['Class'].isnull().sum())
    all_values += species[species['Exporter'] == exporters_10.index[i]]['Class'].notnull().sum()

data = []
for cls in classes_all.index:
    obj = go.Bar(
        x=exporters_10.index,
        y=species[species['Class'] == cls].groupby('Exporter').count()['Year'][exporters_10.index].fillna(0),
        name=cls
    )
    data.append(obj)
    
missing_series = pd.Series(data=missing_values, index=exporters_10.index)
missing_bar = go.Bar(
                x=exporters_10.index,
                y=missing_series,
                name='Unknown',
                marker=dict(
                    color='rgb(230,230,230)',
                ),
            )
data.append(missing_bar)

layout = go.Layout(
    barmode='stack',
    title='Top 10 exporters',
    annotations=[
        dict(
            x=8,
            y=6500,
            xref='x',
            yref='y',
            text='Note! {0}% of data is missing.'.format(str(round(sum(missing_values)/all_values*100, 2))),
            showarrow=False,
        )
    ]
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='top_10_exporters')
importers_10 = species['Importer'].value_counts()[:10]
classes_all = species['Class'].value_counts()

missing_values = []
all_values = 0
for i in range(len(importers_10)):
    missing_values.append(species[species['Importer'] == importers_10.index[i]]['Class'].isnull().sum())
    all_values += species[species['Importer'] == importers_10.index[i]]['Class'].notnull().sum()

data = []
for cls in classes_all.index:
    obj = go.Bar(
        x=importers_10.index,
        y=species[species['Class'] == cls].groupby('Importer').count()['Year'][importers_10.index].fillna(0),
        name=cls
    )
    data.append(obj)

missing_series = pd.Series(data=missing_values, index=importers_10.index)
missing_bar = go.Bar(
                x=importers_10.index,
                y=missing_series,
                name='Unknown',
                marker=dict(
                    color='rgb(230,230,230)',
                ),
            )
data.append(missing_bar)

layout = go.Layout(
    barmode='stack',
    title='Top 10 importers',
    annotations=[
        dict(
            x=8,
            y=9000,
            xref='x',
            yref='y',
            text='Note! {0}% of data is missing.'.format(str(round(sum(missing_values)/all_values*100, 2))),
            showarrow=False,
        )
    ]
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='top_10_importers')
images_species = dict({
    'Acipenser baerii': 'https://drive.google.com/uc?id=15Bun-kuxZEygZk0gAwKfmSflkC3amPoZ',
    'Alligator mississippiensis': 'https://drive.google.com/uc?id=1ArT7yJX6k82vQoIOmlZa1sO5Iv0EZeSm',
    'Ara ararauna': 'https://drive.google.com/uc?id=1USmb_2cGkiaEtQnIyxW45iK2g6iebuWP',
    'Caiman crocodilus yacare': 'https://drive.google.com/uc?id=1gCbh-0wifQo6rNRiPA8yuGQoaIFBkYxJ',
    'Crocodylus niloticus': 'https://drive.google.com/uc?id=1uEGKHwfwhPphhY2rdLgB0xflNMYRc0dJ',
    'Crocodylus novaeguineae': 'https://drive.google.com/uc?id=15JYMo4kfGJdeP6b8aXDt4VOZOK6XU9EL',
    'Crocodylus porosus': 'https://drive.google.com/uc?id=12KH-pAOFoNHX1zRkkhknOy96Due7t5s6',
    'Crocodylus siamensis': 'https://drive.google.com/uc?id=1P4h6IJtHu-yEBrNWIiTaG63F-wtxysPd',
    'Dalbergia nigra': 'https://drive.google.com/uc?id=1s2SQhG6zP4D7q32MbXzSOKhxNbVLIAGG',
    'Elephantidae spp.': 'https://drive.google.com/uc?id=1cI6u1qxIX8TgsBoWhjXjLBH7-OoF1Wwj',
    'Elephas maximus': 'https://drive.google.com/uc?id=18ggnXUYnq8VsChqhwdfCVKeeNuk5rkdV',
    'Euphyllia ancora': 'https://drive.google.com/uc?id=10KrJo-xc9RXnr0DLtIbIWhFI-mmaamgt',
    'Euphyllia glabrescens': 'https://drive.google.com/uc?id=1HsFsnqAjUuc4IGSaSiiUU2SX6blmCcCW',
    'Falco hybrid': 'https://drive.google.com/uc?id=1v3f5TDQpS4_RPfFRIDBxlorv7SayPNe5',
    'Falco peregrinus': 'https://drive.google.com/uc?id=1pBuMj7FLJiD0hqkPh3s-hKen3U0bXy21',
    'Falco rusticolus': 'https://drive.google.com/uc?id=16HKM2Xfvh93kGC-0doZeyGKbGlLx-LD2',
    'Goniopora spp.': 'https://drive.google.com/uc?id=19Ew74b08u5d9Nim6KqW_emrozvkHmuPI',
    'Loxodonta africana': 'https://drive.google.com/uc?id=1po1x2CN2jS6pjqHYS00KCJP7-ny6JoAB',
    'Macaca fascicularis': 'https://drive.google.com/uc?id=12rFm4ILH0LHakVtLCwVRES7aaNIUbzfI',
    'Montipora spp.': 'https://drive.google.com/uc?id=1hKcC58YSkITQa_VAmGelHqX8k2P-Kzqk',
    'Panthera pardus': 'https://drive.google.com/uc?id=1WMnsWoLzb0Xn4ZZJ_ZcSmy8HqwytZg9H',
    'Phalaenopsis hybrid': 'https://drive.google.com/uc?id=1StiOXTb0s1pObmfIpuDDemUcwfgve5kp',
    'Psittacus erithacus': 'https://drive.google.com/uc?id=1NI3_stznblMZAlANRBeZw3ankxtOLL2F',
    'Python bivittatus': 'https://drive.google.com/uc?id=1xaFiygRs_R1-SzW_-db6kji3En1BtNbG',
    'Python brongersmai': 'https://drive.google.com/uc?id=1l5EN4MRUXqXEKT7n7XNOb9JHk29fuTj3',
    'Python reticulatus': 'https://drive.google.com/uc?id=15rWnUPmJAV1w2qzI5QLX5bUvs6y3ULWA',
    'Varanus niloticus': 'https://drive.google.com/uc?id=10o0C7hzQTo3lhE-023N4fDd_NX0nrYV1',
    'Varanus salvator': 'https://drive.google.com/uc?id=1kBi-7BRpmITk12E6qv9zTBoHSo13TSVk',
    'Vicugna vicugna': 'https://drive.google.com/uc?id=1dicv4_PWPrMLCNO1Thwuv6XFC3r8okGt',
    'Dendrobium hybrid': 'https://drive.google.com/uc?id=1dAB7V5TyeZAP4nX9fLCADs3oJlhwGwTl',
    'Anguilla anguilla': 'https://drive.google.com/uc?id=1WvGorPAc9MRl88TTZoobm8ca1imUFKT-',
    'Bulnesia sarmientoi': 'https://drive.google.com/uc?id=13bK4wHLUYuYDth8x9Pcm7nk8i7mybp0f',
    'Panax quinquefolius': 'https://drive.google.com/uc?id=1rWi7-yszsmjGgTfVJdvdci2yLBCFoHHI',
    'Pterocarpus erinaceus': 'https://drive.google.com/uc?id=1qQLVxahabnFZcAkbKDGLXEQOOCiKJRQZ',
    'Strombus gigas': 'https://drive.google.com/uc?id=1lEiyj3frynajSCaQKU83wPb3_L-Iq_Nh',
    'Euphorbia antisyphilitica': 'https://drive.google.com/uc?id=1RvJMtdsUQqTkQ0PkXheaI0GMB9vpxK3F',
    'Prunus africana': 'https://drive.google.com/uc?id=1xlxSFWEFUrj2b4d5s7AtOlBbb8ZotlIM',
    'Scleractinia spp.': 'https://drive.google.com/uc?id=1opfLXHSrcI23kVYCMJ5-TC-5YnARNbSA',
})

images_flags = dict({
    'AE': 'https://drive.google.com/uc?id=1ORAEQjzBwdIhUkwKlstsGNDJk30lu1Nr',
    'AU': 'https://drive.google.com/uc?id=1oWvDln6DGWhaJb1oWcj7vXkE0sRiZzII',
    'BE': 'https://drive.google.com/uc?id=1iRHpvcK8R3eookqhkGsJRbR7H_TKK13s',
    'CA': 'https://drive.google.com/uc?id=1KAaPnLPKbuw5TUQTNgMMNcS0swMu8kFR',
    'CH': 'https://drive.google.com/uc?id=15GWu2TJ77NYAwvEHKOmadh7Y-0L9W2vt',
    'CN': 'https://drive.google.com/uc?id=1m_ew6K0lgDOkp2fYVo-UPhFXOULj6KF9',
    'DE': 'https://drive.google.com/uc?id=149OEzCXv8hJb1y2jm6ji5WVyXuFTsXmY',
    'DK': 'https://drive.google.com/uc?id=1R9SqvTAQq1yZdX-TaIOe4vcghfKjmTbO',
    'EC': 'https://drive.google.com/uc?id=15qktH9MQXh4SoIpLYPEiYuOENeHYmPPJ',
    'FR': 'https://drive.google.com/uc?id=1l95oqKq45VR19JhkwXd2akSVw8mDaX6N',
    'GB': 'https://drive.google.com/uc?id=1Mi86QDBS_amDH4Hf8OYay0yCx3DhLpv2',
    'GY': 'https://drive.google.com/uc?id=1x59jLemZn9Az4Lc2CilJR_4dqAr7TpYJ',
    'ID': 'https://drive.google.com/uc?id=183aDMkokri1WlPYTpANyBAn9lUCkMBCR',
    'IT': 'https://drive.google.com/uc?id=1W3p0AtG83RQne2VH3McAImXoVx81_DAk',
    'JP': 'https://drive.google.com/uc?id=1Aa2aGJptnPxX6uC6Dw8QEH7F7UK4dS_6',
    'KW': 'https://drive.google.com/uc?id=1uFG-B-xE11s3SWO_WLNRAynhCi0slyy4',
    'NL': 'https://drive.google.com/uc?id=1_HK1by4qq9f43MaPOCf3Syy_YTunWqOq',
    'QA': 'https://drive.google.com/uc?id=1IfoYjTD2d-yVv34hGf0kLtdIa1X7PfeD',
    'SG': 'https://drive.google.com/uc?id=1qlU1HdFbSiSdFyfWhWSRe3XytOY3CsZK',
    'TH': 'https://drive.google.com/uc?id=1KfsR0uBe2_bb2mfCHHGMOcS0CC2MHuuH',
    'US': 'https://drive.google.com/uc?id=1LBcjlAO0uaDTaJhVLL9Yu2spL9LxT4TP',
    'HK': 'https://drive.google.com/uc?id=1OHOlr3swoL_mLS1uOfJQCY3tm6UcohdT',
    'ZA': 'https://drive.google.com/uc?id=1Zu13rBsJmWpWLeQ_MBvCxo2Kvm4_zxRM',
    'ZW': 'https://drive.google.com/uc?id=1Pk__J_PCYO0fQ_vkBn2SYPzJOu6F6GiE',
    'MY': 'https://drive.google.com/uc?id=1z4i8WZ_U1FWUmeWSOCOa4_OmZWOzs5fD',
    'MA': 'https://drive.google.com/uc?id=1SzXvA8po8PpiHGBLP9QGNRncUDYqrHD_',
    'NG': 'https://drive.google.com/uc?id=1OCWEvWFzZzCAKQK-bCxA_EenrbOSDbqp',
    'GH': 'https://drive.google.com/uc?id=1zlLVCo2hrV4cNZ23VXgodyC94qFZUddr',
    'BS': 'https://drive.google.com/uc?id=1c7tdCz1u0EHWu_2Dyw3tSelypaeh4zzA',
    'SL': 'https://drive.google.com/uc?id=1xcuNMf1FxbrTn_5v9PB07Uk6yTIAy8fk',
    'RS': 'https://drive.google.com/uc?id=1-gmuXkY9JqBA5kZCZWhlzfrYuPyRZMEA',
    'KR': 'https://drive.google.com/uc?id=1K4001YjVdYUx_PPJU8ScxLhDAUyLDRKu',
    'TN': 'https://drive.google.com/uc?id=1NeaCjHBSsaJb6CF5b4VPktVYCsd_UKdn',
    'PY': 'https://drive.google.com/uc?id=1-Or3rHbxOSD2oqIyEkjWA2h8ZGiFQ_4s',
    'ES': 'https://drive.google.com/uc?id=1rC_mLR_1JDpGeXrqpiIfxvanK0tJkRVD',
    'GM': 'https://drive.google.com/uc?id=1cQG4Ve2u_LO09-NzwP0va816p_k96Oed',
})
# Helping functions
def get_most_common(items, ispurpose=False, issource=False):
    text = ''
    for i in range(len(items)):
        if i == 4 or items[i] / items.sum() < 0.1: 
            break
            
        cat = items.index[i]
        if ispurpose: cat = purpose_dict[cat]
        if issource: cat = source_dict[cat]
            
        new_text = cat + ' (' + str(round(items[i] / items.sum()*100, 2)) + '%) <br>' 
        text = text + new_text
        
    return text

def get_countries(items):
    text = ''
    for i in range(len(items)):
        if i == 4 or (items[i] / items.sum() < 0.1 and i != 0): 
            break
            
        cat = items.index[i]
        if cat not in images_flags:
            new_text = 'XX ({0}%)'.format(round(items[i] / items.sum()*100, 2))
        else:
            new_text = '<div style="display: flex; margin-bottom: 3px;">\
                        <img src="{0}" title="{1}">'.format(images_flags[cat], cat) + '(' + str(round(items[i] / items.sum()*100, 2)) + '%)</div>' 
        text = text + new_text        
        
    return text

def get_image(name):
    if name in images_species:
        return'<img src="{0}" alt="img" width="100" height="100">'.format(images_species[name])
    else:
        return 'No image'
def taxon_summary(taxon_name, taxon_count):
    
    name_count = taxon_name + '<br>(' + str(taxon_count) + ')'
    
    terms = species[species['Taxon'] == taxon_name]['Term'].value_counts()
    term = get_most_common(terms)
    
    purposes = species[species['Taxon'] == taxon_name]['Purpose'].value_counts()
    purpose = get_most_common(purposes, True)
    
    sources = species[species['Taxon'] == taxon_name]['Source'].value_counts()
    source = get_most_common(sources, False, True)
    
    exporters = species[species['Taxon'] == taxon_name]['Exporter'].value_counts()
    exporter = get_countries(exporters)
    
    importers = species[species['Taxon'] == taxon_name]['Importer'].value_counts()
    importer = get_countries(importers)
    
    return ['<img src={0} alt="img" width="100" height="100">'.format(images_species[taxon_name]), 
            name_count, term, purpose, source, exporter, importer]
taxon_10 = species['Taxon'].value_counts()[:10]
columns=['Image', 'Taxon (traded times)', 'Term', 'Purpose', 'Source', 'Exporters', 'Importers']
taxon_10_df = pd.DataFrame(columns=columns)
        
for i in range(len(taxon_10)):
    taxon_10_df.loc[i] = taxon_summary(taxon_10.index[i], taxon_10[i])
display(HTML(taxon_10_df.to_html(escape=False)))
def live_summary(name, count):
    
    name_count = name + '<br>(' + str(count) + ')'
    
    terms = species[(species['Taxon'] == name)]['Term'].value_counts()
    term = get_most_common(terms)
    
    purposes = species[(species['Taxon'] == name)]['Purpose'].value_counts()
    purpose = get_most_common(purposes, True)
    
    sources = species[species['Taxon'] == name]['Source'].value_counts()
    source = get_most_common(sources, False, True)
        
    exporters = species[(species['Taxon'] == name)]['Exporter'].value_counts()
    exporter = get_countries(exporters)
    
    importers = species[(species['Taxon'] == name)]['Importer'].value_counts()
    importer = get_countries(importers)
    
    return ['<img src="{0}" alt="img" width="100" height="100">'.format(images_species[name]), 
            name_count, term, purpose, source, exporter, importer]
live_10 = species[species['Term'] == 'live']['Taxon'].value_counts()[:10]
columns=['Image', 'Taxon (traded times)', 'Term', 'Purpose', 'Source', 'Exporters', 'Importers']
live_10_df = pd.DataFrame(columns=columns)
        
for i in range(len(live_10)):
    live_10_df.loc[i] = live_summary(live_10.index[i], live_10[i])
display(HTML(live_10_df.to_html(escape=False)))
def app_summary(name, count):
    
    name_count = name + '<br>(' + str(count) + ')'
    
    terms = species[(species['Taxon'] == name)]['Term'].value_counts()
    term = get_most_common(terms)
    
    purposes = species[(species['Taxon'] == name)]['Purpose'].value_counts()
    purpose = get_most_common(purposes, True)
    
    sources = species[species['Taxon'] == name]['Source'].value_counts()
    source = get_most_common(sources, False, True)
        
    exporters = species[(species['Taxon'] == name)]['Exporter'].value_counts()
    exporter = get_countries(exporters)
    
    importers = species[(species['Taxon'] == name)]['Importer'].value_counts()
    importer = get_countries(importers)
    
    return ['<img src="{0}" alt="img" width="100" height="100">'.format(images_species[name]), 
            name_count, term, purpose, source, exporter, importer]
app_10 = species[species['App.'] == 'I']['Taxon'].value_counts()[:10]
columns=['Image', 'Taxon (traded times)', 'Term', 'Purpose', 'Source', 'Exporters', 'Importers']
app_10_df = pd.DataFrame(columns=columns)
        
for i in range(len(app_10)):
    app_10_df.loc[i] = app_summary(app_10.index[i], app_10[i])
display(HTML(app_10_df.to_html(escape=False)))
# Preprocess quantities
unit_weights = [('g', 1e3), ('mg', 1e6), ('microgrammes', 1e9)]

for unit_weight in unit_weights:
    indeces = species['Unit'] == unit_weight[0]
    species.loc[indeces, ['Importer reported quantity', 'Exporter reported quantity']] = \
                species.loc[indeces, ['Importer reported quantity', 'Exporter reported quantity']] / unit_weight[1]
    species.loc[indeces, 'Unit'] = 'kg'
    
def kg_summary(name, count):
    
    name_count = name + '<br>(' + str(int(count/1000)) + ' tonnes)'
    
    terms = species[(species['Taxon'] == name)]['Term'].value_counts()
    term = get_most_common(terms)
    
    purposes = species[(species['Taxon'] == name)]['Purpose'].value_counts()
    purpose = get_most_common(purposes, True)
    
    sources = species[species['Taxon'] == name]['Source'].value_counts()
    source = get_most_common(sources, False, True)
        
    exporters = species[(species['Taxon'] == name)]['Exporter'].value_counts()
    exporter = get_countries(exporters)
    
    importers = species[(species['Taxon'] == name)]['Importer'].value_counts()
    importer = get_countries(importers)
    
    image = get_image(name)
    
    return [image, name_count, term, purpose, source, exporter, importer]
kg_10 = species[species['Unit'] == 'kg'].groupby('Taxon').sum()[['Importer reported quantity', 'Exporter reported quantity']]\
    .max(axis=1).sort_values(ascending=False)[:10]
    
columns=['Image', 'Taxon (traded amount)', 'Term', 'Purpose', 'Source', 'Exporters', 'Importers']
kg_10_df = pd.DataFrame(columns=columns)
        
for i in range(len(app_10)):
    kg_10_df.loc[i] = kg_summary(kg_10.index[i], kg_10[i])
    
display(HTML(kg_10_df.to_html(escape=False)))