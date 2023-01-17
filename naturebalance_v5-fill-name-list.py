import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#所有输入文件为utf8编码，如果不是，记事本打开另存为utf8编码。

#把输入文件放在桌面上，这里是输入“法院名称”的列表文件（注意：只要法院名称，其他多余的行和列都用excel都删了）

inputNameList=pd.read_csv("../input/list.csv")



nameList=inputNameList

nameList["省"]="NULL"

nameList["市"]="NULL"

nameList["县"]="NULL"

nameList.head()
#把输入文件放在桌面上，这里是输入“地区名称”的列表文件（注意：把文件开头多余的行用excel删了）

#如果区域名称的县这一列的县级市有定义缺失，程序会把相近的填充，解决办法是，把这个没有定义的县级市补上。

#如何差错，写在输出这一段

inputRegion=pd.read_csv("../input/region.csv")



region=inputRegion.loc[:,['province','city','town']]

region=region[~region.town.isnull()]

region.head()
#这个要花时间，原理就是对于每一个法院名称，都用地区名字表的最后一列来匹配，只要匹配上，就自动填充。

for i in range(len(nameList)):

    print(i)

    for j in range(len(region)):

        if region.iloc[j,2] in nameList.iloc[i,0]:

            if nameList.iloc[i,3]=='NULL':

                nameList.iloc[i,3]=region.iloc[j,2]

                nameList.iloc[i,2]=region.iloc[j,1]

                nameList.iloc[i,1]=region.iloc[j,0]

                print(region.iloc[j,2]+" "+nameList.iloc[i,0]+" "+nameList.iloc[i,3]+" "+nameList.iloc[i,2]+" "+nameList.iloc[i,1])                

            else:

                if nameList.iloc[i,2] in nameList.iloc[i,0]:

                    pass

                else:

                    if region.iloc[j,1] in nameList.iloc[i,0]:                    

                        nameList.iloc[i,3]=region.iloc[j,2]

                        nameList.iloc[i,2]=region.iloc[j,1]

                        nameList.iloc[i,1]=region.iloc[j,0]

                    else:                 

                        if (len(region.iloc[j,2])>len(nameList.iloc[i,3])):

                            nameList.iloc[i,3]=region.iloc[j,2]

                            nameList.iloc[i,2]=region.iloc[j,1]

                            nameList.iloc[i,1]=region.iloc[j,0]

                        else:

                            pass

                print(region.iloc[j,2]+" "+nameList.iloc[i,0]+" "+nameList.iloc[i,3]+" "+nameList.iloc[i,2]+" "+nameList.iloc[i,1]) 

        else:pass            
nameList.head(100)
region_1=inputRegion.loc[:,['province','city']]

region_1=region_1.drop_duplicates('city','first')

region_1=region_1[~region_1.city.isnull()]

region_1.head()
#这个也要花时间，但是比前面的少很多，原理一样，只是匹配名称列表倒数第二列，但是多了一个裁剪功能，也就是“上海市”和“上海”这两个都算能匹配上。

for i in range(len(nameList)):

    for j in range(len(region_1)):

        if region_1.iloc[j,1] in nameList.iloc[i,0]:

            nameList.iloc[i,2]=region_1.iloc[j,1]

            nameList.iloc[i,1]=region_1.iloc[j,0]

            print(region_1.iloc[j,1],nameList.iloc[i,0])

        else:

            if region_1.iloc[j,1][:-1] in nameList.iloc[i,0]:

                nameList.iloc[i,2]=region_1.iloc[j,1]

                nameList.iloc[i,1]=region_1.iloc[j,0]           

                print(region_1.iloc[j,1],nameList.iloc[i,0])

  
nameList.head(100)
#这里是输出文件，默认保存在桌面。名字随便取。

nameList.to_csv("output.csv")
#以下是差错检验。

#如果区域名称的县这一列的县级市有定义缺失，程序会把相近的填充，解决办法是，把这个没有定义的县级市在输入的区域名称文件中补上。

#这种错误主要出现在县这一列里面两个字的匹配，字数多的一般不会匹配错误。

nameList['c' ]=0
for i in range(len(nameList)):

    nameList.iloc[i,4]=len(nameList.iloc[i,3])

    
nameList.sort_values(by=['c','县'],ascending= True) 

#比如 揭阳市揭东区人民法院 如果揭东区 这个县级城市没有定义，那么就会匹配 东区

#解决办法就是在输入的区域名称列表里补充这个县级城市的名字。

#以下是原来没有定义缺失的。

#广东省 揭阳市 揭东区  （源文件是 揭东县 增加 揭东区 如果删除揭东县 那么万一以后出现就无法匹配了 ）

#浙江省 温州市 鹿城区  （原文件 缺失 增加 鹿城区）

#广东省 广州市 增城区  （源文件是 增城市 增加 增城区）

#
