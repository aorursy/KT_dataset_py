import pandas as pd

import numpy as np

import math

import folium

from IPython.display import HTML, display
zhtm = ["https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Little_Brown_Bat_with_White_Nose_Syndrome_%28Greeley_Mine%2C_cropped%29.jpg/400px-Little_Brown_Bat_with_White_Nose_Syndrome_%28Greeley_Mine%2C_cropped%29.jpg",

        "https://wildlife.ca.gov/portals/0/Images/WIL/WSN/Bat-Visible-Growth.jpg",

        "https://wildlife.ca.gov/Portals/0/Images/WIL/WSN/AffectedBats.jpg",

        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Pseudogymnoascus_destructans_grey_culture.jpg/440px-Pseudogymnoascus_destructans_grey_culture.jpg"]



zhtm = ''.join(["<img style='height:180px; margin:0px; float:left; border:5px solid green;' src='%s' />" % str(s) for s in zhtm])



display(HTML(zhtm))
zhtm=["https://apps.npr.org/dailygraphics/graphics/coronavirus-spread-map-20200122/img/_world-Artboard_1.jpg",

     "https://s3.us-west-2.amazonaws.com/prod-is-cms-assets/wns/prod/e95b2410-21ab-11ea-a154-67ca1cde5e5d-WNSSpreadMap_8_30_2019.jpg"]



zhtm = ''.join(["<img style='height:270px; margin:0px; float:left; border:5px solid green;' src='%s' />" % str(s) for s in zhtm])



display(HTML(zhtm))
dfC = pd.DataFrame(

data = [["Italy",41.9,12.6,92472,10023],["Spain",40.5,-3.7,72335,5820],["Germany",51.2,10.5,57695,433],["France",46.2,2.2,37575,2314],["Iran",32.4,53.7,35408,2517],["UK",55.4,-3.4,17089,1019],["Switzerland",46.8,8.2,14076,264],["Netherlands",52.1,5.3,9762,639],["S Korea",35.9,127.8,9478,144],["Belgium",50.5,4.5,9134,353],["Austria",47.5,14.6,8188,68],["Turkey",39,35.2,7402,108],["Canada",56.1,-106.3,5576,60],["Portugal",39.4,-8.2,5170,100],["Norway",60.5,8.5,4012,23],["Brazil",-14.2,-51.9,3904,111],["Australia",-25.3,133.8,3635,14],["Israel",31,34.9,3619,12],["Sweden",60.1,18.6,3447,105],["Czechia",49.8,15.5,2541,11],["Ireland",53.4,-8.2,2415,36],["Malaysia",4.2,102,2320,27],["Denmark",56.3,9.5,2201,65],["Chile",-35.7,-71.5,1909,6],["Luxembourg",49.8,6.1,1831,18],["Ecuador",-1.8,-78.2,1823,48],["Japan",36.2,138.3,1693,52],["Poland",51.9,19.1,1638,18],["Pakistan",30.4,69.3,1495,12],["Romania",45.9,25,1452,34],["Russia",61.5,105.3,1264,4],["Thailand",15.9,101,1245,6],["Saudi Arabia",23.9,45.1,1203,4],["South Africa",-30.6,22.9,1187,1],["Finland",61.9,25.7,1167,9],["Indonesia",-0.8,113.9,1155,102],["Philippines",12.9,121.8,1075,68],["Greece",39.1,21.8,1061,32],["India",20.6,79,987,24],["Iceland",65,-19,963,2],["Singapore",1.4,103.8,802,2],["Panama",8.5,-80.8,786,14],["Dominican Republic",18.7,-70.2,719,28],["Mexico",23.6,-102.6,717,12],["Diamond Princess",35.4,139.7,712,10],["Argentina",-38.4,-63.6,690,18],["Slovenia",46.2,15,684,9],["Peru",-9.2,-75,671,16],["Serbia",44,21,659,10],["Croatia",45.1,15.2,657,5],["Estonia",58.6,25,645,1],["Colombia",4.6,-74.3,608,6],["Qatar",25.4,51.2,590,1],["Egypt",26.8,30.8,576,36],["Hong Kong",22.4,114.1,560,4],["Iraq",33.2,43.7,506,42],["Bahrain",25.9,50.6,476,4],["UAE",23.4,53.8,468,2],["Algeria",28,1.7,454,29],["New Zealand",-40.9,174.9,451,0],["Lebanon",33.9,35.9,412,8],["Armenia",40.1,45,407,1],["Lithuania",55.2,23.9,394,7],["Morocco",31.8,-7.1,390,25],["Ukraine",48.4,31.2,356,9],["Hungary",47.2,19.5,343,11],["Bulgaria",42.7,25.5,331,6],["Andorra",42.5,1.6,308,3],["Latvia",56.9,24.6,305,0],["Costa Rica",9.7,-83.8,295,2],["Slovakia",48.7,19.7,292,0],["Taiwan",23.7,121,283,2],["Uruguay",-32.5,-55.8,274,0],["Bosnia and Herzegovina",43.9,17.7,258,5],["Tunisia",33.9,9.5,257,8],["Jordan",30.6,36.2,246,1],["North Macedonia",42,-21.3,241,4],["Kuwait",29.3,47.5,235,0],["Moldova",47.4,28.4,231,2],["Kazakhstan",48,66.9,228,1],["San Marino",43.9,12.5,224,22],["Burkina Faso",12.2,-1.6,207,11],["Albania",41.2,20.2,197,10],["Réunion",-21.1,55.5,183,0],["Azerbaijan",40.1,47.6,182,4],["Cyprus",35.1,33.4,179,5],["Vietnam",14.1,108.3,174,0],["Faeroe Islands",56.3,9.5,155,0],["Oman",21.5,55.9,152,0],["Malta",35.9,14.4,149,0],["Ghana",7.9,-1,141,5],["Senegal",14.5,-14.5,130,0],["Brunei",4.5,114.7,120,1],["Cuba",21.5,-77.8,119,3],["Sri Lanka",7.9,80.8,113,1],["Venezuela",6.4,-66.6,113,2],["Afghanistan",33.9,67.7,110,4],["Uzbekistan",41.4,64.6,104,2],["Guadeloupe",17,-62.1,102,2],["Mauritius",-20.3,57.6,102,2],["Ivory Coast",6.5,5.2,101,0],["China/ Hubei",23.7,108.8,67800,3122],["China/ Guangdong",24.5,101.3,1370,8],["China/ Henan",26.5,117.9,1273,22],["China/ Zhejiang",30.3,102.8,1232,1],["China/ Hunan",35.4,109.2,1018,4],["China/ Anhui",47.1,128.7,990,6],["China/ Jiangxi",31.2,121.4,935,1],["China/ Shandong",37.9,114.9,761,7],["China/ Jiangsu",29.4,106.9,631,0],["China/ Chongqing",39.3,117.4,576,6],["China/ Sichuan",35.8,104.3,540,3],["China/ Heilongjiang",35.9,117.9,483,13],["China/ Beijing",30.7,112.2,479,8],["China/ Shanghai",33.1,119.8,363,3],["China/ Hebei",34.3,113.4,318,6],["China/ Fujian",40.2,116.2,296,1],["China/ Guangxi",19.6,109.9,253,2],["China/ Shaanxi",42.5,87.5,246,3],["China/ Yunnan",41.9,122.5,176,2],["China/ Hainan",26.8,107.3,168,6],["China/ Guizhou",37.2,111.9,146,2],["China/ Tianjin",23.4,113.8,136,3],["China/ Gansu",43.4,115.1,133,2],["China/ Shanxi",29.1,119.8,133,0],["China/ Liaoning",30.6,117.9,125,1],["China/ Jilin",27.1,114.9,93,1],["China/ Xinjiang",30.2,88.8,76,3],["China/ Neimenggu",27.6,111.9,75,1],["China/ Ningxia",35.7,96.4,75,0],["China/ Qinghai",37.2,106.2,18,0],["China/ Xizang",43.2,126.4,1,0],["USA/ New York",43,-75,52318,728],["USA/ New Jersey",39.8,-74.9,11124,140],["USA/ California",36.8,-119.4,4980,104],["USA/ Michigan",44.2,-84.5,4650,111],["USA/ Massachusetts",42.4,-71.4,4257,44],["USA/ Florida",28,-81.8,3763,54],["USA/ Washington",47.8,-120.7,3723,175],["USA/ Illinois",40,-89,3491,47],["USA/ Louisiana",30.4,-92.3,3315,137],["USA/ Pennsylvania",41.2,-77.2,2751,34],["USA/ Georgia",33.2,-83.4,2366,69],["USA/ Texas",31,-100,2329,30],["USA/ Colorado",39.1,-105.4,1734,31],["USA/ Tennessee",35.9,-86.7,1512,7],["USA/ Ohio",40.4,-83,1406,25],["USA/ Connecticut",41.6,-72.7,1291,27],["USA/ Indiana",40.3,-86.1,1232,31],["USA/ Maryland",39,-76.6,992,5],["USA/ Wisconsin",44.5,-89.5,989,15],["USA/ North Carolina",35.8,-80.8,952,4],["USA/ Missouri",38.6,-92.6,838,10],["USA/ Arizona",34,-111.1,773,15],["USA/ Virginia",37.9,-78,739,17],["USA/ Alabama",32.3,-86.9,696,4],["USA/ Mississippi",33,-90,663,13],["USA/ South Carolina",33.8,-81.2,660,15],["USA/ Nevada",39.9,-117.2,621,10],["USA/ Utah",39.4,-112,602,2],["USA/ Oregon",44,-120.5,479,13],["USA/ Minnesota",46.4,-94.6,441,5],["USA/ Arkansas",34.8,-92.2,404,3],["USA/ Kentucky",37.8,-84.3,394,8],["USA/ Oklahoma",36.1,-96.9,377,15],["USA/ District of Columbia",43,-75,304,4],["USA/ Iowa",42,-93.6,298,3],["USA/ Kansas",38.5,-98,261,5],["USA/ Rhode Island",41.7,-71.5,239,2],["USA/ Idaho",44.1,-114.7,230,4],["USA/ Delaware",39,-75.5,213,3],["USA/ Maine",45.4,-69,211,1],["USA/ Vermont",44,-72.7,211,12],["USA/ New Mexico",34.3,-106,208,1],["USA/ New Hampshire",44,-71.5,187,2],["USA/ Montana",47,-109.5,129,1],["USA/ Hawaii",19.7,-155.8,120,0],["USA/ Puerto Rico",18.2,-66.6,100,3],["USA/ Nebraska",41.5,-100,96,2],["USA/ West Virginia",39,-80.5,96,1],["USA/ Alaska",66.2,-153.4,85,2],["USA/ North Dakota",47.7,-100.4,83,1],["USA/ Wyoming",43.1,-107.3,82,0],["USA/ South Dakota",44.5,-100,68,1],["USA/ Guam",13.4,144.8,51,1],["USA/ Virgin Islands",18.3,-64.9,21,0]],

columns = ['Region','Lat','Lng','CovCases','CovDeaths'])



dfC = dfC.sort_values(by=['CovCases','CovDeaths','Lat'], ascending=False)



def fbin(i):

    return 2 * int(i/2) + 1



dfC['LatB'] = dfC.apply(lambda x: fbin(x['Lat']), axis=1)

dfC['LngB'] = dfC.apply(lambda x: fbin(x['Lng']), axis=1)



pd.options.display.float_format = '{:,.0f}'.format

dfC.head()
dfW = pd.DataFrame(

data = [["Belgium",50.8,4.5,10],["Canada/ New Brunswick",46.5,-66,9],["Canada/ Nova Scotia",45,-63,9],["Canada/ Ontario",49.3,-84.5,10],["Canada/ Prince Edward Island",46.4,-63.3,7],["Canada/ Quebec",52,-72,10],["China/ Beijing",39.9,116.4,5],["China/ Jilin",43.6,126.2,5],["China/ Liaoning",41.3,122.7,5],["China/ Shandong",36.3,118.3,5],["Croatia",45.2,15.5,7],["Czechia",49.8,15,10],["Estonia",59,26,10],["France",46,2,10],["Germany",51.5,10.5,10],["Hungary",47,20,10],["Latvia",57,25,6],["Luxembourg",49.8,6.2,5],["Netherlands",52.3,5.8,10],["Poland",52,20,10],["Russia",60,100,6],["Slovakia",48.7,19.5,10],["Slovenia",46.1,15,6],["Switzerland",47,8,10],["UK",54.8,-2.7,7],["Ukraine",49,32,10],["USA/ Alabama",32.8,-86.8,8],["USA/ Arkansas",34.8,-92.5,8],["USA/ Connecticut",41.7,-72.7,12],["USA/ Delaware",39,-75.5,10],["USA/ Georgia",32.8,-83.5,7],["USA/ Illinois",40,-89.3,7],["USA/ Indiana",40,-86.3,9],["USA/ Iowa",42,-93.5,8],["USA/ Kentucky",38.2,-84.9,9],["USA/ Maine",45.5,-69.2,9],["USA/ Maryland",39,-76.7,10],["USA/ Massachusetts",42.4,-71.1,12],["USA/ Michigan",44.3,-85.5,6],["USA/ Minnesota",46.3,-94.3,8],["USA/ Mississippi",32.8,-89.8,6],["USA/ Missouri",38.3,-92.5,10],["USA/ Nebraska",41.5,-99.8,5],["USA/ New Hampshire",43.7,-71.5,11],["USA/ New Jersey",40.2,-74.5,11],["USA/ New York",43,-75.5,12],["USA/ North Carolina",35.5,-80,9],["USA/ Ohio",40.3,-83,9],["USA/ Oklahoma",35.5,-97.5,5],["USA/ Pennsylvania",40.3,-76.9,10],["USA/ Rhode Island",41.8,-71.5,4],["USA/ South Carolina",34,-81,7],["USA/ Tennessee",35.8,-86.3,10],["USA/ Texas",31.3,-99.3,3],["USA/ Vermont",44,-72.7,12],["USA/ Virginia",37.5,-77.4,11],["USA/ Washington",47.5,-120.5,4],["USA/ West Virginia",38.5,-80.5,11],["USA/ Wisconsin",44.5,-90,6]],

columns = ['Region','Lat', 'Lng', 'WnsYears'])



dfW = dfW.sort_values(by=['WnsYears','Lat'], ascending=False)



dfW['LatB'] = dfW.apply(lambda x: fbin(x['Lat']), axis=1)

dfW['LngB'] = dfW.apply(lambda x: fbin(x['Lng']), axis=1)



# pd.options.display.float_format = '{:,.0f}'.format

dfW.head()
dfCg = dfC[['LatB','LngB','CovCases','CovDeaths']].groupby(['LatB','LngB']).sum().reset_index()

dfWg = dfW[['LatB','LngB','WnsYears']].groupby(['LatB','LngB']).max().reset_index()



dfCW = pd.merge(dfCg, dfWg, on=['LatB','LngB'], how='outer', indicator=False)

dfCW = dfCW.fillna(0)



dfCW['CovCas'] = dfCW.apply(lambda x: int(math.log(x['CovCases']+9,10)), axis=1)

dfCW['CovDth'] = dfCW.apply(lambda x: int(math.log(x['CovDeaths']+9,10)), axis=1)

dfCW['WnsYrs'] = dfCW.apply(lambda x: int(math.log(x['WnsYears']+1,2)), axis=1)



dfCW = dfCW.sort_values(by=['CovCases','CovDeaths','LatB'], ascending=False)



pd.options.display.float_format = '{:,.0f}'.format

dfCW.head()
zmap = folium.Map(location=[0, 0], zoom_start=2)



colorC = {0:'yellow', 1:'orange', 2:'coral', 3:'tomato', 4:'red'}



for la,lo,cca,cda,wya,cc,cd,wy in zip(dfCW['LatB'],dfCW['LngB'],dfCW['CovCases'],dfCW['CovDeaths'],dfCW['WnsYears'],dfCW['CovCas'],dfCW['CovDth'],dfCW['WnsYrs']):

    

    if cc > 0:

        folium.CircleMarker(

            [la, lo],

            radius = 3*cc+1,

            popup = str(cca) + '<br>' + str(cda) + '<br>' + str(wya) ,

            color = 'b',

            key_on = cc,

            threshold_scale = [0,1,2,3,4],

            fill_color = colorC[cc],

            fill = True,

            fill_opacity = 1

            ).add_to(zmap)

    

    if wy > 0:

        colorW = ['black','grean','green','blue','blue'][wy]

        folium.PolyLine([[la-1/2,lo-1/2],[la+1/2,lo+1/2]], color=colorW).add_to(zmap)

        folium.PolyLine([[la-1/2,lo+1/2],[la+1/2,lo-1/2]], color=colorW).add_to(zmap)

        

zmap
def corInd(c,w):

    if c>0 and w>0:

        return 4 - abs(c - w)

    else:

        return 0
dfCW['CorrIndex'] = dfCW.apply(lambda x: corInd(x['CovCas'], x['WnsYrs']), axis=1)

dfCW['Bins'] = 1



dfCW.head()
dfCor = dfCW[['CorrIndex','Bins']].groupby(['CorrIndex']).sum().reset_index()

dfCor['BinsPerc'] = 100*dfCor['Bins']/dfCW['Bins'].sum()



dfDict = pd.DataFrame(data=[[4,"Striking"],[3,"Noticeable"],[2,"Somewhat"],[1,"Faint"],[0,"None"]],

                        columns=['CorrIndex','Correlation'])



dfCor = pd.merge(dfCor, dfDict, on=['CorrIndex'], how='left', indicator=False)

dfCor = dfCor.sort_values(by=['CorrIndex'], ascending=False)



dfCor