from selenium import webdriver

import time
driver = webdriver.Chrome('../input/chromedriver')

driver.get("http://info.nec.go.kr/main/showDocument.xhtml?electionId=0020170509&topMenuId=VC&secondMenuId=VCCP08")
import pandas as pd

import numpy as np



import platform

import matplotlib.pyplot as plt



%matplotlib inline



from matplotlib import font_manager, rc

if platform.system() == 'Darwin':

    rc('font', family='AppleGothic')

elif platform.system() == 'Windows':

    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()

    rc('font', family=font_name)

else:

    print('Unknown system... sorry~~~~')    



plt.rcParams['axes.unicode_minus'] = False
from bs4 import BeautifulSoup


sido_name_list = []

sigun_name_list = []

pop = []

moon = []

hong = []

ahn = []



sido_name_values = ['서울특별시','부산광역시','대구광역시','인천광역시','광주광역시','대전광역시','울산광역시','세종특별자치시','경기도','강원도','충청북도','충청남도','전라북도','전라남도','경상북도','경상남도','제주특별자치도']

for sido_value in sido_name_values:

    element = driver.find_element_by_id("cityCode")

    element.send_keys(sido_value)

    

    time.sleep(1)

    

    sigun_list_raw = driver.find_element_by_xpath("""//*[@id="townCode"]""")

    sigun_list = sigun_list_raw.find_elements_by_tag_name("option")



    sigun_names_values = [option.text for option in sigun_list]

    sigun_names_values = sigun_names_values[1:]

    

    for sigun_value in sigun_names_values:

        element = driver.find_element_by_id("townCode")

        element.send_keys(sigun_value)

        

        time.sleep(1)

        

        driver.find_element_by_xpath("""//*[@id="spanSubmit"]/input""").click()

        

        time.sleep(1)

        

        html = driver.page_source

        soup = BeautifulSoup(html, "lxml")

        

        tmp = soup.find_all('td', 'alignR')

        tmp_values = [float(tmp_val.get_text().replace(',', '')) for tmp_val in tmp[1:5]]

        

        sido_name_list.append(sido_value)

        sigun_name_list.append(sigun_value)

        pop.append(tmp_values[0])

        moon.append(tmp_values[1])

        hong.append(tmp_values[2])

        ahn.append(tmp_values[3])

        

        time.sleep(1)
import re
election_result = pd.DataFrame({'광역시도':sido_name_list, '시군':sigun_name_list, 'pop':pop, 

                                'moon':moon, 'hong':hong, 'ahn':ahn})



ID = []



for n in election_result.index:

    if (election_result['광역시도'][n][-1] == '시') & (election_result['광역시도'][n] != '세종특별자치시'):

        if len(election_result['시군'][n]) == 2:

            ID.append(election_result['광역시도'][n][:2] + ' ' + election_result['시군'][n])

        else:

            ID.append(election_result['광역시도'][n][:2] + ' ' + election_result['시군'][n][:-1])

            

    elif (election_result['광역시도'][n][-1] == '도'):

        tmp = election_result['시군'][n]

        

        if tmp[0] not in ['시','군']:

            tmp2 = re.split('시|군', tmp)

        else:

            tmp2 = [tmp[:-1], '']

        

        if len(tmp2[1]) == 2:

            tmp3 = tmp2[0] + ' ' + tmp2[1]

        elif len(tmp2[1]) >= 3:

            tmp3 = tmp2[0] + ' ' + tmp2[1][:-1]

        else:

            tmp3 = tmp2[0]

            

        ID.append(tmp3)

        

    else:

        ID.append('세종')



election_result['ID'] = ID
election_result['rate_moon'] = election_result['moon'] / election_result['pop'] * 100

election_result['rate_hong'] = election_result['hong'] / election_result['pop'] * 100

election_result['rate_ahn'] = election_result['ahn'] / election_result['pop'] * 100
draw_korea = pd.read_csv('./draw_korea_data.csv', encoding='utf-8', index_col=0)



election_result.loc[125, 'ID'] = '고성(강원)'

election_result.loc[233, 'ID'] = '고성(경남)'



election_result.loc[228, 'ID'] = '창원 합포'

election_result.loc[229, 'ID'] = '창원 회원'



ahn_tmp = election_result.loc[85, 'ahn']/3

hong_tmp = election_result.loc[85, 'hong']/3

moon_tmp = election_result.loc[85, 'moon']/3

pop_tmp = election_result.loc[85, 'pop']/3



rate_moon_tmp = election_result.loc[85, 'rate_moon']

rate_hong_tmp = election_result.loc[85, 'rate_hong']

rate_ahn_tmp = election_result.loc[85, 'rate_ahn']



election_result.loc[250] = [ahn_tmp, hong_tmp, moon_tmp, pop_tmp, 

                           '경기도', '부천시', '부천 소사', 

                           rate_moon_tmp, rate_hong_tmp, rate_ahn_tmp]

election_result.loc[251] = [ahn_tmp, hong_tmp, moon_tmp, pop_tmp, 

                           '경기도', '부천시', '부천 오정', 

                           rate_moon_tmp, rate_hong_tmp, rate_ahn_tmp]

election_result.loc[252] = [ahn_tmp, hong_tmp, moon_tmp, pop_tmp, 

                           '경기도', '부천시', '부천 원미', 

                           rate_moon_tmp, rate_hong_tmp, rate_ahn_tmp]



election_result.drop([85], inplace=True)
set(draw_korea['ID'].unique()) - set(election_result['ID'].unique())
final_elect_data = pd.merge(election_result, draw_korea, how='left', on=['ID'])



final_elect_data['moon_vs_hong'] = final_elect_data['rate_moon'] - final_elect_data['rate_hong']

final_elect_data['moon_vs_ahn'] = final_elect_data['rate_moon'] - final_elect_data['rate_ahn']

final_elect_data['ahn_vs_hong'] = final_elect_data['rate_ahn'] - final_elect_data['rate_hong']
def drawKorea(targetData, blockedMap, cmapname):

    gamma = 0.75



    whitelabelmin = 20.



    datalabel = targetData



    vmin = -50

    vmax = 50



    BORDER_LINES = [

        [(5, 1), (5,2), (7,2), (7,3), (11,3), (11,0)], # 인천

        [(5,4), (5,5), (2,5), (2,7), (4,7), (4,9), (7,9), 

         (7,7), (9,7), (9,5), (10,5), (10,4), (5,4)], # 서울

        [(1,7), (1,8), (3,8), (3,10), (10,10), (10,7), 

         (12,7), (12,6), (11,6), (11,5), (12, 5), (12,4), 

         (11,4), (11,3)], # 경기도

        [(8,10), (8,11), (6,11), (6,12)], # 강원도

        [(12,5), (13,5), (13,4), (14,4), (14,5), (15,5), 

         (15,4), (16,4), (16,2)], # 충청북도

        [(16,4), (17,4), (17,5), (16,5), (16,6), (19,6), 

         (19,5), (20,5), (20,4), (21,4), (21,3), (19,3), (19,1)], # 전라북도

        [(13,5), (13,6), (16,6)], # 대전시

        [(13,5), (14,5)], #세종시

        [(21,2), (21,3), (22,3), (22,4), (24,4), (24,2), (21,2)], #광주

        [(20,5), (21,5), (21,6), (23,6)], #전라남도

        [(10,8), (12,8), (12,9), (14,9), (14,8), (16,8), (16,6)], #충청북도

        [(14,9), (14,11), (14,12), (13,12), (13,13)], #경상북도

        [(15,8), (17,8), (17,10), (16,10), (16,11), (14,11)], #대구

        [(17,9), (18,9), (18,8), (19,8), (19,9), (20,9), (20,10), (21,10)], #부산

        [(16,11), (16,13)], #울산

    #     [(9,14), (9,15)], 

        [(27,5), (27,6), (25,6)],

    ]



    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)

    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)

    

    plt.figure(figsize=(9, 11))

    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, edgecolor='#aaaaaa', linewidth=0.5)



    # 지역 이름 표시

    for idx, row in blockedMap.iterrows():

        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. (중구, 서구)

        if len(row['ID'].split())==2:

            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])

        elif row['ID'][:2]=='고성':

            dispname = '고성'

        else:

            dispname = row['ID']



        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.

        if len(dispname.splitlines()[-1]) >= 3:

            fontsize, linespacing = 10.0, 1.1

        else:

            fontsize, linespacing = 11, 1.



        annocolor = 'white' if np.abs(row[targetData]) > whitelabelmin else 'black'

        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',

                     fontsize=fontsize, ha='center', va='center', color=annocolor,

                     linespacing=linespacing)



    # 시도 경계 그린다.

    for path in BORDER_LINES:

        ys, xs = zip(*path)

        plt.plot(xs, ys, c='black', lw=2)



    plt.gca().invert_yaxis()



    plt.axis('off')



    cb = plt.colorbar(shrink=.1, aspect=10)

    cb.set_label(datalabel)



    plt.tight_layout()

    plt.show()
drawKorea('moon_vs_hong', final_elect_data, 'RdBu')
drawKorea('moon_vs_ahn', final_elect_data, 'RdBu')
drawKorea('ahn_vs_hong', final_elect_data, 'RdBu')