import pandas as pd

import numpy as np

import json

from xgboost import XGBClassifier

import holidays

from datetime import datetime

import category_encoders as ce

import matplotlib.pyplot as plt

%matplotlib inline
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

pd.set_option('display.width', 1000)

train_df = pd.read_csv('../input/klps-creditscring-challenge-for-students/train.csv')

test_df = pd.read_csv('../input/klps-creditscring-challenge-for-students/test.csv')
with open('../input/added-dataset/tinhhuyen.json', 'r') as openfile:

    diaChinhDict = json.load(openfile)

    

with open('../input/added-dataset/DATATinh.json', 'r') as openfile:  

    DATA = json.load(openfile) 

colname = DATA["ha noi"].keys()



with open('../input/added-dataset/QuanHuyenDict.json', 'r') as openfile:  

    quanhuyenDict = json.load(openfile) 
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'

s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

def remove_accents(x):

    if str(x) != 'nan':

        input_str = str(x).lower()

        s = ''

        for c in input_str:

            if c in s1:

                s += s0[s1.index(c)]

            else:

                s += c

        s = s.replace(" province", "").replace(" city", "")

        return s

    else:

        return x



def convertProvince(x):

    row = remove_accents(x)

    if str(row) != 'nan':

        state = row

        if str(row).lower() == 'hanoi':

            state = "ha noi"

        elif str(row).lower() == 'haiphong':

            state = "hai phong"

        elif str(row).lower() == 'thua thien–hue':

            state = "hue"

        elif str(row).lower() == 'ba ria–vung tau':

            state = "vung tau"

        else:

            state = row

        if state not in diaChinhDict:

            return 100

        else:

            return diaChinhDict[state]['MaTP']

    else:

        return np.nan





train_df['currentLocationState_code'] = train_df['currentLocationState'].apply(convertProvince)

train_df['homeTownState_code'] = train_df['homeTownState'].apply(convertProvince)

test_df['currentLocationState_code'] = test_df['currentLocationState'].apply(convertProvince)

test_df['homeTownState_code'] = test_df['homeTownState'].apply(convertProvince)

cols = train_df['homeTownState_code'].value_counts().index.tolist()

print(cols)
MaTPDict = {}

for state in diaChinhDict:

    MaTPDict[diaChinhDict[state]['MaTP']] = state

MaTPDict
def gdp(df):

    for col in ["currentLocationState_code","homeTownState_code"]:

        for name in colname:

            df[col+"_"+name] = np.nan

        for i in range(len(df.index)):

            if not pd.isna(df[col][i]) and df[col][i]<100:

                for name in colname:

                    #print (col+"_"+name, i, DATA[df[col + "_TinhThanhpho"][i]][name])

                    df[col+"_"+name][i] = DATA[MaTPDict[df[col][i]]][name] 
gdp(train_df)

gdp(test_df)
columns_state = ["id", "currentLocationState_code", "homeTownState_code"]

for name in colname:

    columns_state.append('currentLocationState_code_' + name)

    columns_state.append('homeTownState_code_' + name)



trainLocation = train_df[columns_state]

testLocation = test_df[columns_state]

trainLocation
def encoding_gdp(train_gdp, test_gdp):

    gdp_cols = [col for col in train_gdp.columns ]    

    df_gdp = train_gdp.append(test_gdp, ignore_index=True)



    data = df_gdp[gdp_cols]



    for col in gdp_cols:

        data[col] = data[col].fillna(-1)



    _train = data[data['id'] < 53030]

    _test = data[data['id'] >= 53030]

    

    return _train, _test
trainLocation_EncodeOnly, testLocation_EncodeOnly = encoding_gdp(trainLocation, testLocation)
SOURCE_CHARACTERS = ["A","Á","À","Ả","Ã","Ạ","a","á","à","ả","ã","ạ",

"Ă","Ắ","Ằ","Ẳ","Ẵ","Ặ","ă","ắ","ằ","ẳ","ẵ","ặ",

"Â","Ấ","Ầ","Ẩ","Ẫ","Ậ","â","ấ","ầ","ẩ","ẫ","ậ",

"E","É","È","Ẻ","Ẽ","Ẹ","e","é","è","ẻ","ẽ","ẹ",

"Ê","Ế","Ề","Ể","Ễ","Ệ","ê","ế","ề","ể","ễ","ệ",

"I","Í","Ì","Ỉ","Ĩ","Ị","i","í","ì","ỉ","ĩ","ị",

"U","Ú","Ù","Ủ","Ũ","Ụ","u","ú","ù","ủ","ũ","ụ",

"Ư","Ứ","Ừ","Ử","Ữ","Ự","ư","ứ","ừ","ử","ữ","ự",

"Y","Ý","Ỳ","Ỷ","Ỹ","Ỵ","y","ý","ỳ","ỷ","ỹ","ỵ",

"O","Ó","Ò","Ỏ","Õ","Ọ","o","ó","ò","ỏ","õ","ọ",

"Ô","Ố","Ồ","Ổ","Ỗ","Ộ","ô","ố","ồ","ổ","ỗ","ộ",

"Ơ","Ớ","Ờ","Ở","Ỡ","Ợ","ơ","ớ","ờ","ở","ỡ","ợ",

"Đ","đ"]

DESTINATION_CHARACTERS = ['A', 'A', 'A', 'A', 'A', 'A', 'a', 'a', 'a', 'a', 'a', 'a',

'A', 'A', 'A', 'A', 'A', 'A', 'a', 'a', 'a', 'a', 'a', 'a',

'A', 'A', 'A', 'A', 'A', 'A', 'a', 'a', 'a', 'a', 'a', 'a',

'E', 'E', 'E', 'E', 'E', 'E', 'e', 'e', 'e', 'e', 'e', 'e',

'E', 'E', 'E', 'E', 'E', 'E', 'e', 'e', 'e', 'e', 'e', 'e',

'I', 'I', 'I', 'I', 'I', 'I', 'i', 'i', 'i', 'i', 'i', 'i',

'U', 'U', 'U', 'U', 'U', 'U', 'u', 'u', 'u', 'u', 'u', 'u',

'U', 'U', 'U', 'U', 'U', 'U', 'u', 'u', 'u', 'u', 'u', 'u',

'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'y', 'y', 'y', 'y', 'y', 'y',

'O', 'O', 'O', 'O', 'O', 'O', 'o', 'o', 'o', 'o', 'o', 'o',

'O', 'O', 'O', 'O', 'O', 'O', 'o', 'o', 'o', 'o', 'o', 'o',

'O', 'O', 'O', 'O', 'O', 'O', 'o', 'o', 'o', 'o', 'o', 'o',

'D', 'd']

Dict = {}

for i in range(len(SOURCE_CHARACTERS)):

    Dict[SOURCE_CHARACTERS[i]] = DESTINATION_CHARACTERS[i]
def RemoveAccent(String):

    StringRemoveAccented = []

    if type(String) != str:

        return np.nan

    for i in range(len(String)):

        if String[i] in Dict:

            StringRemoveAccented.append(Dict[String[i]])

        else:

            StringRemoveAccented.append(String[i])

    return "".join(StringRemoveAccented).lower()
labels = train_df['label']

df = train_df[["id","diaChi","Field_48","Field_49"]].append(test_df[["id","diaChi","Field_48","Field_49"]], ignore_index=True)
df["diaChiRemoveAccent"] = df["diaChi"].apply(RemoveAccent)

df["Field_48RemoveAccent"] = df["Field_48"].apply(RemoveAccent)

df["Field_49RemoveAccent"] = df["Field_49"].apply(RemoveAccent)
df = df.drop(["diaChi","Field_48","Field_49"],axis = 1)
SizeProcess = len(df.index)

phraseDict = {

                    "hn" : "ha noi",

                    "hcm" : "ho chi minh",

                    "sai gon": "ho chi minh",

                    "q.": "quan ",

                    "q1":"quan 1",

                    "q2":"quan 2",

                    "q3":"quan 3",

                    "q4":"quan 4",

                    "q5":"quan 5",

                    "q6":"quan 6",

                    "q7":"quan 7",

                    "q8":"quan 8",

                    "q9":"quan 9",

                    "quy nhon": "qui nhon"

                }
for col in ["diaChi","Field_48","Field_49"]:

    df[col+"_TinhThanhpho"] = np.nan

    df[col+"_QuanHuyen"] = np.nan

    df[col+"_PhuongXa"] = np.nan

    df[col+"_MaTP"] = np.nan

    df[col+"_MaQH"] = np.nan

    df[col+"_MaPX"] = np.nan

    for i in range(SizeProcess):

        string = df[col+"RemoveAccent"][i]

        if type(string) != str:

            continue

        else:

            for phrase in list(phraseDict.keys()):

                string = string.replace(phrase,phraseDict[phrase])

#             print(i,"    ",string)

            leng = 10

            while (leng <= len(string)+10 and pd.isna(df[col+"_TinhThanhpho"][i])):

                for province in list(diaChinhDict.keys()):

                    #print(province,"       ",string[-leng:])

                    if province in string[-leng:]:

                        df[col+"_TinhThanhpho"][i] = province

                        df[col+"_MaTP"][i] = diaChinhDict[province]["MaTP"]

                        string = string.replace(province,"",1)

                        while (leng <= len(string)+10 and pd.isna(df[col+"_QuanHuyen"][i])):

                            for district in list(diaChinhDict[province]["quanhuyen"].keys()):

                                #print(district,"       ",string[-leng:])

                                if district in string[-leng:]:

                                    df[col+"_QuanHuyen"][i] = district

                                    df[col+"_MaQH"][i] = diaChinhDict[province]["quanhuyen"][district]["MaQH"]

                                    string = string.replace(district,"",1)

                                    while (leng <= len(string)+10 and pd.isna(df[col+"_PhuongXa"][i])):

                                        for ward in diaChinhDict[province]["quanhuyen"][district]["phuongxa"]:

                                            #print(ward["NamePX"],"       ",string[-leng:])

                                

                                            if ward["NamePX"] in string[-leng:]:

                                                df[col+"_PhuongXa"][i] = ward["NamePX"]

                                                df[col+"_MaPX"][i] = ward["MaPX"]

                                                break

                                        leng+=10

                                    break

                            leng+=10

                        break

                leng+=10
for col in ["diaChi","Field_48","Field_49"]: #

    for i in range(SizeProcess):

        if pd.isna(df[col+"_QuanHuyen"][i]):

            string = df[col+"RemoveAccent"][i]

            if type(string) != str:

                continue

            else:

                for phrase in list(phraseDict.keys()):

                    string = string.replace(phrase,phraseDict[phrase])

#                 print(i,"    ",string)

                leng = 10

                while (leng <= len(string)+10 and pd.isna(df[col+"_QuanHuyen"][i])):

                    for district in list(quanhuyenDict.keys()):

                        if district in string[-leng:]:

                            df[col+"_QuanHuyen"][i] = district

                            string = string.replace(district,"",1)

                            while (leng <= len(string)+10 and pd.isna(df[col+"_PhuongXa"][i])):

                                for ward in quanhuyenDict[district]["phuongxa"]:

                                    if ward in string[-leng:]:

                                        df[col+"_PhuongXa"][i] = ward

                                        df[col+"_MaPX"][i] = quanhuyenDict[district]["phuongxa"][ward]["MaPX"]

                                        df[col+"_MaQH"][i] = quanhuyenDict[district]["phuongxa"][ward]["MaQH"]

                                        df[col+"_TinhThanhpho"][i] = quanhuyenDict[district]["phuongxa"][ward]["thanhPho"]

                                        df[col+"_MaTP"][i] = quanhuyenDict[district]["phuongxa"][ward]["MaTP"]

                                        break

                                leng+=10

                            if(pd.isna(df[col+"_TinhThanhpho"][i])):

                                df[col+"_TinhThanhpho"][i] = quanhuyenDict[district]["thanhPho"]

                                df[col+"_MaTP"][i] = quanhuyenDict[district]["MaTP"]

                            if(pd.isna(df[col+"_MaQH"][i])):

                                df[col+"_MaQH"][i] = quanhuyenDict[district]["MaQH"]

                            break

                    leng+=10
for col in ["diaChi","Field_48","Field_49"]:

    for name in colname:

        df[col+"_"+name] = np.nan

    for i in range(SizeProcess):

        if type(df[col + "_TinhThanhpho"][i]) == str:

            for name in colname:

                #print (col+"_"+name, i, DATA[df[col + "_TinhThanhpho"][i]][name])

                df[col+"_"+name][i] = DATA[df[col + "_TinhThanhpho"][i]][name] 
num_features = [col for col in df.columns if df[col].dtype != "object"]

final_df = df[num_features].fillna(-1)
trainDiaChi_Field48_Field49 = final_df[final_df['id'] < 53030]

testDiaChi_Field48_Field49 = final_df[final_df['id'] >= 53030]    

trainDiaChi_Field48_Field49["label"] = labels
def build_model(train, test):

    labels = train['label']

    df = train.drop(columns=['label']).append(test, ignore_index=True)



    remove_features = ['Field_1', 'Field_2', 'Field_4', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9',

                       'Field_11', 'Field_12', 'Field_15', 'Field_18', 'Field_25', 'Field_32', 'Field_33',

                       'Field_34', 'Field_35', 'gioiTinh', 'diaChi', 'Field_36', 'Field_40',

                       'Field_43', 'Field_44', 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',

                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',

                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',

                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',

                       'homeTownCountry', 'homeTownName', 'homeTownState', 'F_startDate', 'F_endDate',

                       'E_startDate', 'E_endDate', 'C_startDate', 'C_endDate', 'G_startDate', 'G_endDate',

                       'A_startDate', 'A_endDate', 'brief']



    cat_features_count_encode = ['Field_4', 'Field_12', 'Field_18', 'Field_34', 'diaChi', 'gioiTinh', 'Field_36',

                                 'Field_45', 'Field_46', 'Field_47', 'Field_48', 'Field_49',

                       'Field_54', 'Field_55', 'Field_56', 'Field_61', 'Field_62', 'Field_65', 'Field_66',

                       'Field_68', 'maCv', 'info_social_sex', 'data.basic_info.locale', 'currentLocationCity',

                       'currentLocationCountry', 'currentLocationName', 'currentLocationState', 'homeTownCity',

                       'homeTownCountry', 'homeTownName', 'homeTownState', 'brief']

    

    cat_features_target_encode = ['Field_4', 'Field_12', 'gioiTinh', 'Field_36', 'Field_47', 'Field_55',

                                  'Field_61', 'Field_62', 'Field_65', 'Field_66', 'info_social_sex', 'brief']

    

    cat_features_onehot_encode = ['Field_4', 'Field_12', 'gioiTinh', 'Field_47', 'Field_62', 'Field_66',

                                  'info_social_sex', 'maCv', 'Field_56', 'Field_61']

    

    cat_one_two = ['Field_47']

    cat_I_II = ['Field_62']



    int_missing_features = ['Field_38']



    num_features = [col for col in df.columns if df[col].dtype != "object" and col not in ['id', 'label'] ]

    

#     unless_features = []



    #---------------------------- Data cleaning ----------------------------

    data = df

#     data = df.drop(columns=unless_features, inplace=True)

    

    cat_date_array = ['Field_1', 'Field_2', 'Field_5', 'Field_6', 'Field_7', 'Field_8', 'Field_9', 'Field_11',

                      'Field_15', 'Field_25', 'Field_32', 'Field_33', 'Field_35', 'Field_40', 'Field_43',

                      'Field_44', 'F_startDate', 'F_endDate', 'E_startDate', 'E_endDate', 'C_startDate',

                      'C_endDate', 'G_startDate', 'G_endDate', 'A_startDate', 'A_endDate']

    cat_date_time_array = ['Field_1', 'Field_2', 'Field_43', 'Field_44']

    for col in cat_date_array:

        data[col+'Year'] = pd.DatetimeIndex(data[col]).year

        data[col+'Month'] = pd.DatetimeIndex(data[col]).month

        data[col+'Day'] = pd.DatetimeIndex(data[col]).day

    for col in cat_date_time_array:

        data[col+'Hour'] = pd.DatetimeIndex(data[col]).hour

        

        



    # Hàm xử lý các cột có thông tin dạng Năm-tháng-ngày Giờ-phút-giây

    def convert_datetime_hour(x):

        if type(x) == float: # nếu giá trị là null, giữ nguyên là null

            return x

        else:

            try:

                return datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ') # các giá trị không có microsecond 

            except:

                return datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ') # các giá trị có microsecond 



    # Hàm xử lý các cột có thông tin dạng Năm-tháng-ngày

    def convert_datetime(x):

        if type(x) == float: # nếu giá trị là null, giữ nguyên là null

            return x

        else:

            try:

                return datetime.strptime(x, '%Y-%m-%d')

            except:

                return datetime.strptime(x, '%m/%d/%Y')



    def convert(df, cols, f):

        for col in cols:

            df[col+'_converted'] = df[col].apply(lambda x: f(x))



    vn_holidays = holidays.VN()

    def isHoliday(x):

        if type(x) == pd._libs.tslibs.nattype.NaTType:

            return x

        else:

            return 1 if x in vn_holidays else 0



    set1 = ['Field_1', 'Field_2', 'Field_43', 'Field_44']

    convert(data, set1, convert_datetime_hour)



     # Convert columns Y-m-d

    set2 = ['Field_5', 'Field_6', 'Field_7', 'Field_8','Field_9','Field_11','Field_15','Field_25', 

               'Field_32','Field_33','Field_35','Field_40',

               'F_startDate','F_endDate','E_startDate','E_endDate','C_startDate','C_endDate','G_startDate',

                'G_endDate','A_startDate','A_endDate']

    convert(data, set2, convert_datetime)





    for col in cat_date_array:

        column = data[col+'_converted']

        remove_features.append(col+'_converted')

        # Create weekend columns

        data[col+'_weekend'] = column.apply(lambda x: 1 if x.weekday() > 4 else 0)

        # Create quarter columns

        data[col+'_quarter'] = column.apply(lambda x: (x.month-1)//3 + 1)

        # Create holiday columns

        data[col+'_holiday'] = column.apply(lambda x: isHoliday(x))

        num_features.append(col+'_weekend')

        num_features.append(col+'_quarter')

        num_features.append(col+'_holiday')

    



    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'

    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

    def remove_accents(x):

        if str(x) != 'nan':

            input_str = str(x).lower()

            s = ''

            for c in input_str:

                if c in s1:

                    s += s0[s1.index(c)]

                else:

                    s += c

            s = s.replace(" ", "").replace("-", "").replace(",", "")

            return s

        else:

            return x

        

    def ngaySinhYear(row):

        if str(row) != 'nan':

            data_str = str(row)

            return int(data_str[:4])

        else:

            return row

    data['ngaySinhYear'] = data['ngaySinh'].apply(ngaySinhYear)



    def ngaySinhMonth(row):

        if str(row) != 'nan':

            data_str = str(row)

            return int(data_str[4:6])

        else:

            return row

    data['ngaySinhMonth'] = data['ngaySinh'].apply(ngaySinhMonth)



    def ngaySinhDay(row):

        if str(row) != 'nan':

            data_str = str(row)

            return int(data_str[6:8])

        else:

            return row

    data['ngaySinhDay'] = data['ngaySinh'].apply(ngaySinhDay)

    

    data['currentLocationCity'] = data['currentLocationCity'].apply(remove_accents)

    data['currentLocationName'] = data['currentLocationName'].apply(remove_accents)

    data['currentLocationState'] = data['currentLocationState'].apply(remove_accents)

    data['homeTownCity'] = data['homeTownCity'].apply(remove_accents)

    data['homeTownState'] = data['homeTownState'].apply(remove_accents)

    data['homeTownName'] = data['homeTownName'].apply(remove_accents)



    def to_lower_case(item):

        return item.map(lambda x: x if type(x)!=str else x.lower())



    def replace_data(typ, data, keys, values):

        for index, value in enumerate(data):

            if str(value) != 'nan':

                if typ == 1:

                    for key in keys:

                        if key in str(value): 

                            data[index] = values

                else:

                    ok = True

                    for key in keys:

                        if key in str(value): 

                            ok = False

                    if ok:

                        data[index] = values



    data["maCv"] = to_lower_case(data['maCv'])

    replace_data(1, data["maCv"],['công nhân', 'cn', 'thợ', 'gia công', 'đóng gói', 'sản phẩm', 'may', 'sửa chữa', 'may', 'lao động', 'ld', 'thủ công'],'cn')

    replace_data(1, data["maCv"],['nhân viên', 'nv'],'nv')

    replace_data(1, data["maCv"],['giáo viên', 'giảng viên', 'gv'],'gv')

    replace_data(1, data["maCv"],['cán bộ', 'cán sự', 'bí thư', 'chuyên viên', 'giám đốc', 'cb', 'gd', 'trưởng', 'phó', 'quản lý', 'ql', 'cb'],'cb')

    replace_data(1, data["maCv"],['kế toán', 'bán', 'đại diện kinh doanh', 'kiểm ngân', 'thủ kho', 'thủ quỹ', 'giao dịch viên', 'thu ngân', 'kt', 'kinh doanh', 'kt'],'kt')

    replace_data(1, data["maCv"],['kỹ sư', 'kỹ thuật', 'kĩ thuật', 'ks'],'ks')

    replace_data(1, data["maCv"],['phục vụ', 'phụ', 'tạp vụ', 'cô nuôi', 'pv', 'lễ tân', 'phu'],'phu')

    replace_data(1, data["maCv"],['y sỹ', 'y sĩ', 'dược', 'cấp dưỡng', 'điều dưỡng', 'hộ lý', 'y tá', 'thú y', 'yte'],'yte')

    replace_data(1, data["maCv"],['lái xe', 'tài xế', 'lx'],'lx')

    replace_data(1, data["maCv"],['bảo mẫu', 'giám sát'],'bm')

    replace_data(1, data["maCv"],['huấn luyện', 'hướng dẫn', 'trợ lý', 'thư ký', 'tiếp viên', 'hd'],'hd')

    replace_data(0, data["maCv"],['cn', 'nv', 'gv', 'cb', 'kt', 'ks', 'phu', 'yte', 'lx', 'bm', 'hd'],'other')

    

    data["Field_56"] = to_lower_case(data['Field_56'])

    data["Field_61"] = to_lower_case(data['Field_61'])

    for col in ['Field_56', 'Field_61']:

        replace_data(1, data[col],['ngoài quốc doanh', 'nqd'],'nqd')

        replace_data(1, data[col],['hành chánh sự nghiệp', 'sự nghiệp', 'hcsn'],'hcsn')

        replace_data(1, data[col],['doanh nghiệp nhà nước', 'dnnn'],'dnnn')

        replace_data(1, data[col],['nghèo', 'khó khăn', 'hưu trí', 'bảo trợ', 'thất nghiệp'],'ngheo')

        replace_data(1, data[col],['học sinh', 'sinh viên', 'trẻ em', 'hs'],'hs')

        replace_data(1, data[col],['nông', 'nn'],'nn')

        replace_data(1, data[col],['công lập', 'cl'],'cl')

        replace_data(1, data[col],['cán bộ', 'đại biểu'],'cb')

        replace_data(1, data[col],['chiến binh', 'vũ trang', 'kháng chiến', 'vt'],'vt')

        replace_data(1, data[col],['đoàn', 'đảng', 'doandoi'],'doandoi')

        replace_data(1, data[col],['phường', 'xã'],'pxh')

        replace_data(1, data[col],['hợp tác xã'],'htx')

        replace_data(1, data[col],['hộ'],'ho')

        replace_data(1, data[col],['khối'],'khoi')

        replace_data(1, data[col],['thân nhân'],'thannhan')

        replace_data(0, data[col],['doandoi', 'ct', 'cb', 'cl', 'nn', 'hs', 'ngheo', 'dnnn', 'hcsn', 'nqd', 'pxh', 'htx', 'ho', 'khoi', 'thanhnhan'], 'other')



    # cols_with_missing = [col for col in data.columns if data[col].isnull().any()]

    # for col in cols_with_missing:

    #     data[col + '_was_missing'] = data[col].isnull()

    #     data[col + '_was_missing'] = data[col + '_was_missing'].map(lambda x: 1 if x == True else 0)



    for col in num_features:

        if data[col].dtype == "float64":

            data[col] = data[col].fillna(-1.0)

        else:

            data[col] = data[col].fillna(-1)

    for col in remove_features:

        data[col] = data[col].fillna("Missing")



    count_en = ce.CountEncoder()

    cat_ce = count_en.fit_transform(data[cat_features_count_encode])

    data = data.join(cat_ce.add_suffix("_ce"))

    

    cat_oh = pd.get_dummies(data[cat_features_onehot_encode])

    data = data.join(cat_oh.add_suffix("_oh"))



    for col in cat_I_II:

        data[col + '_encode'] = data[col].map(

            lambda x: 4 if x == "IV" else (

                1 if x == "I" else (

                    2 if x == "II" else (

                        3 if x == "III" else (

                            # 5 if x == "V" else np.nan

                            5 if x == "V" else ( -999 if x == "Missing" else -1)

                        ) ) ) ) )

        data[col] = data[col].map(lambda x: "Missing" if x != "I" and x != "II" and x != "III" and x != "IV" and x != "V"  else x)



    for col in cat_one_two:

        data[col + '_encode'] = data[col].map(lambda x: 4 if x == "Four" else (

                1 if x == "One" else (

                    2 if x == "Two" else (

                        3 if x == "Three" else (

                            # 0 if x == "Zero" or x == "Zezo" else np.nan

                            0 if x == "Zero" or x == "Zezo" else ( -999 if x == "Missing" else -1)

                        ) ) ) ) )

    

    for col in int_missing_features:

        tmp = data[col].map(lambda x: -1 if x == "None" else x)

        data[col] = pd.to_numeric(tmp, errors='coerce').fillna(-999)

        # data[col] = pd.to_numeric(tmp, errors='coerce')



    _train = data[data['id'] < 53030]

    _test = data[data['id'] >= 53030]

    

    _train["label"] = labels

    

    target_enc = ce.TargetEncoder(cols=cat_features_target_encode)

    target_enc.fit(_train[cat_features_target_encode], _train['label'])

    _train = _train.join(target_enc.transform(_train[cat_features_target_encode]).add_suffix('_target'))

    _test = _test.join(target_enc.transform(_test[cat_features_target_encode]).add_suffix('_target'))

    

#     cb_enc = ce.CatBoostEncoder(cols=cat_features_target_encode, random_state=7)

#     cb_enc.fit(_train[cat_features_target_encode], _train['label'])

#     _train = _train.join(cb_enc.transform(_train[cat_features_target_encode]).add_suffix('_cb'))

#     _test = _test.join(cb_enc.transform(_test[cat_features_target_encode]).add_suffix('_cb'))

    

    _train.drop(columns=remove_features, inplace=True)

    _test.drop(columns=remove_features, inplace=True)

    

    return _train, _test
train_data, test_data = build_model(train_df, test_df)
merge_train_data = pd.concat([train_data, trainLocation_EncodeOnly, trainDiaChi_Field48_Field49], axis=1)

merge_test_data = pd.concat([test_data.reset_index().drop(columns=["index"]), testLocation_EncodeOnly, testDiaChi_Field48_Field49], axis=1)
merge_train_data = merge_train_data.loc[:,~merge_train_data.columns.duplicated()]

merge_test_data = merge_test_data.loc[:,~merge_test_data.columns.duplicated()]
def calculate_woe_iv(dataset, feature, target):

    lst = []

    for i in range(dataset[feature].nunique()):

        val = list(dataset[feature].unique())[i]

        lst.append({

            'Value': val,

            'All': dataset[dataset[feature] == val].count()[feature],

            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],

            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]

        })

    dset = pd.DataFrame(lst)

    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()

    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()

    dset['WoE'] = np.log(dset['Distr_Good'] / dset['Distr_Bad'])

    dset = dset.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

    iv = dset['IV'].sum()

    dset = dset.sort_values(by='WoE')

    return dset, iv



def woe(_df):

    USELESS_PREDICTOR = []

    WEAK_PREDICTOR = []

    MEDIUM_PREDICTOR = []

    STRONG_PREDICTOR = []

    GOOD_PREDICTOR = []

    for col in _df.columns:

        if col == 'label' or col == 'id': continue

        else:

            print('WoE and IV for column: {}'.format(col))

            final, iv = calculate_woe_iv(_df, col, 'label')

    #         print(final)

            iv = round(iv,2)

            print('IV score: ' + str(iv))

            print('\n')

            if (iv < 0.02) and col not in USELESS_PREDICTOR:

                USELESS_PREDICTOR.append(col)

            elif iv >= 0.02 and iv < 0.1 and col not in WEAK_PREDICTOR:

                WEAK_PREDICTOR.append(col)

            elif iv >= 0.1 and iv < 0.3 and col not in MEDIUM_PREDICTOR:

                MEDIUM_PREDICTOR.append(col)

            elif iv >= 0.3 and iv < 0.5 and col not in STRONG_PREDICTOR:

                STRONG_PREDICTOR.append(col)

            elif iv >= 0.5 and col not in GOOD_PREDICTOR:

                GOOD_PREDICTOR.append(col)



    return USELESS_PREDICTOR, WEAK_PREDICTOR, MEDIUM_PREDICTOR, STRONG_PREDICTOR, GOOD_PREDICTOR
USELESS_PREDICTOR, WEAK_PREDICTOR, MEDIUM_PREDICTOR, STRONG_PREDICTOR, GOOD_PREDICTOR = woe(merge_train_data)
IGNORE_FEATURE = USELESS_PREDICTOR

final_train_data = merge_train_data.drop(columns=IGNORE_FEATURE)

final_test_data = merge_test_data.drop(columns=[col for col in IGNORE_FEATURE if col not in ['label']])
import gc

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, recall_score, classification_report

import seaborn as sns



# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')

    

def display_roc_curve(y_, oof_preds_,sub_preds_,folds_idx_):

    # Plot ROC curves

    plt.figure(figsize=(6,6))

    scores = [] 

    for n_fold, (_, val_idx) in enumerate(folds_idx_):  

        # Plot the roc curve

        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])

        score = 2 * auc(fpr, tpr) -1

#         score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])

        scores.append(score)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (Gini = %0.4f)' % (n_fold + 1, score))

    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)

    score = 2 * auc(fpr, tpr) -1

#     score = roc_auc_score(y_, oof_preds_)

    plt.plot(fpr, tpr, color='b',

             label='Avg ROC (Gini = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),

             lw=2, alpha=.8)

    

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('LightGBM ROC Curve')

    plt.legend(loc="lower right")

    plt.tight_layout()

    

    plt.savefig('roc_curve.png')
def xgboost_model(train_df, test_df, num_folds, stratified = False):

# Divide in training/validation and test data

    print("Starting xbgboost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    gc.collect()

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=500)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=500)

    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['label','id']]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['label'])):        

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['label'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['label'].iloc[valid_idx]



        # RandomForest parameters found by Bayesian optimization

        clf = XGBClassifier(learning_rate=0.1, gamma=0.2, n_estimators=140, max_depth=8,

                        min_child_weight=39.3259775, subsample=0.8715623, colsample_bytree=1.0,

                        reg_alpha=0.041545473,

                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)



        clf.fit(train_x, train_y)



        oof_pred = clf.predict_proba(valid_x)[:, 1]

        

        pred = clf.predict(valid_x)

        print('F1 Score: ' + str( f1_score(valid_y, pred) ))

        print('Recall Score: ' + str( recall_score(valid_y, pred) ))

        

        sub_pred = clf.predict_proba(test_df[feats])[:, 1] / folds.n_splits

        oof_preds[valid_idx] = oof_pred

        sub_preds += sub_pred

                

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full AUC score %.6f' % roc_auc_score(train_df['label'], oof_preds))

    

    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_df[feats], train_df['label'])]

    display_roc_curve(y_=train_df['label'],oof_preds_=oof_preds,sub_preds_ = sub_preds, folds_idx_=folds_idx)



    test_df['label'] = sub_preds

    test_df[['id', 'label']].to_csv('./xgboost.csv', index= False)

    display_importances(feature_importance_df)

    return feature_importance_df
xgboost_model(final_train_data, final_test_data, 5, True)