import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train = pd.read_csv("../input/train_1.csv").fillna(0)

days = [r for r in range(train.iloc[2,1:].shape[0])]



def plot_views(idx):

    fig = plt.figure(1,figsize=(10,5))

    plt.plot(days, train.iloc[idx,1:])

    plt.xlabel('day')

    plt.ylabel('views')

    plt.title(train.iloc[[idx],0])

    plt.show()

    

def plot_trend(idx, trend):

    fig = plt.figure(1,figsize=(10,5))

    plt.plot(days, trend)

    plt.xlabel('day')

    plt.ylabel('trend')

    plt.title(train.iloc[[idx],0])

    plt.show()
# for Page B1A4, If anyone has and idea on how can I use the files from github directly please tell me.

page_20_json = {"0":32,"1":30,"2":34,"3":40,"4":42,"5":27,"6":32,"7":30,"8":24,"9":26,"10":39,"11":31,"12":27,"13":31,"14":29,"15":33,"16":33,"17":38,"18":33,"19":32,"20":34,"21":32,"22":27,"23":39,"24":27,"25":48,"26":45,"27":41,"28":37,"29":32,"30":46,"31":45,"32":41,"33":39,"34":42,"35":42,"36":45,"37":56,"38":66,"39":100,"40":99,"41":77,"42":84,"43":71,"44":67,"45":76,"46":72,"47":59,"48":54,"49":58,"50":56,"51":56,"52":62,"53":62,"54":55,"55":48,"56":46,"57":47,"58":49,"59":55,"60":55,"61":44,"62":44,"63":45,"64":47,"65":57,"66":66,"67":55,"68":37,"69":33,"70":33,"71":39,"72":43,"73":47,"74":58,"75":38,"76":38,"77":40,"78":34,"79":45,"80":50,"81":47,"82":43,"83":38,"84":41,"85":38,"86":42,"87":50,"88":48,"89":34,"90":32,"91":33,"92":38,"93":32,"94":43,"95":38,"96":30,"97":28,"98":28,"99":29,"100":34,"101":42,"102":39,"103":32,"104":31,"105":30,"106":26,"107":31,"108":35,"109":44,"110":29,"111":29,"112":29,"113":34,"114":39,"115":36,"116":41,"117":29,"118":34,"119":33,"120":30,"121":29,"122":35,"123":42,"124":32,"125":30,"126":29,"127":30,"128":33,"129":36,"130":44,"131":40,"132":33,"133":29,"134":38,"135":38,"136":32,"137":38,"138":32,"139":35,"140":38,"141":33,"142":38,"143":41,"144":45,"145":33,"146":24,"147":31,"148":27,"149":28,"150":39,"151":37,"152":27,"153":31,"154":25,"155":23,"156":29,"157":33,"158":30,"159":24,"160":27,"161":34,"162":40,"163":34,"164":39,"165":34,"166":31,"167":29,"168":32,"169":30,"170":30,"171":31,"172":31,"173":27,"174":28,"175":37,"176":40,"177":37,"178":44,"179":46,"180":34,"181":28,"182":36,"183":42,"184":50,"185":47,"186":42,"187":32,"188":28,"189":39,"190":29,"191":29,"192":34,"193":31,"194":35,"195":28,"196":28,"197":30,"198":30,"199":32,"200":38,"201":30,"202":30,"203":28,"204":30,"205":27,"206":40,"207":36,"208":31,"209":28,"210":25,"211":28,"212":31,"213":40,"214":42,"215":26,"216":28,"217":31,"218":29,"219":32,"220":33,"221":35,"222":29,"223":28,"224":28,"225":25,"226":26,"227":30,"228":38,"229":28,"230":36,"231":36,"232":32,"233":29,"234":39,"235":33,"236":30,"237":28,"238":25,"239":25,"240":25,"241":32,"242":34,"243":26,"244":19,"245":23,"246":21,"247":26,"248":28,"249":32,"250":25,"251":23,"252":21,"253":27,"254":21,"255":30,"256":29,"257":27,"258":25,"259":15,"260":17,"261":20,"262":23,"263":25,"264":18,"265":16,"266":20,"267":19,"268":17,"269":25,"270":84,"271":100,"272":67,"273":56,"274":57,"275":55,"276":54,"277":50,"278":36,"279":38,"280":36,"281":32,"282":41,"283":43,"284":45,"285":35,"286":37,"287":30,"288":35,"289":36,"290":39,"291":38,"292":28,"293":32,"294":26,"295":30,"296":31,"297":32,"298":31,"299":27,"300":25,"301":27,"302":26,"303":23,"304":31,"305":52,"306":55,"307":58,"308":88,"309":96,"310":64,"311":71,"312":66,"313":58,"314":62,"315":66,"316":81,"317":59,"318":58,"319":60,"320":59,"321":63,"322":86,"323":70,"324":50,"325":49,"326":56,"327":59,"328":53,"329":71,"330":61,"331":57,"332":37,"333":56,"334":53,"335":46,"336":66,"337":59,"338":56,"339":51,"340":47,"341":52,"342":63,"343":76,"344":77,"345":49,"346":51,"347":44,"348":53,"349":61,"350":64,"351":55,"352":59,"353":44,"354":45,"355":60,"356":44,"357":59,"358":60,"359":44,"360":43,"361":44,"362":47,"363":46,"364":55,"365":57,"366":49,"367":33,"368":35,"369":40,"370":54,"371":59,"372":49,"373":42,"374":43,"375":45,"376":58,"377":58,"378":53,"379":65,"380":54,"381":45,"382":47,"383":52,"384":46,"385":65,"386":57,"387":48,"388":52,"389":38,"390":47,"391":45,"392":55,"393":56,"394":40,"395":38,"396":54,"397":46,"398":47,"399":64,"400":50,"401":39,"402":48,"403":52,"404":48,"405":48,"406":51,"407":61,"408":41,"409":38,"410":36,"411":40,"412":53,"413":54,"414":52,"415":42,"416":43,"417":38,"418":37,"419":39,"420":45,"421":43,"422":38,"423":41,"424":36,"425":39,"426":36,"427":46,"428":51,"429":36,"430":34,"431":36,"432":45,"433":44,"434":46,"435":39,"436":35,"437":31,"438":38,"439":34,"440":40,"441":51,"442":48,"443":49,"444":42,"445":48,"446":50,"447":42,"448":55,"449":56,"450":43,"451":37,"452":40,"453":37,"454":36,"455":53,"456":58,"457":47,"458":39,"459":38,"460":39,"461":30,"462":46,"463":35,"464":31,"465":51,"466":59,"467":42,"468":52,"469":53,"470":55,"471":43,"472":51,"473":62,"474":54,"475":51,"476":61,"477":76,"478":52,"479":62,"480":66,"481":56,"482":56,"483":55,"484":78,"485":53,"486":57,"487":56,"488":64,"489":100,"490":91,"491":74,"492":67,"493":53,"494":48,"495":54,"496":53,"497":71,"498":70,"499":50,"500":59,"501":54,"502":55,"503":47,"504":80,"505":72,"506":58,"507":62,"508":58,"509":54,"510":62,"511":66,"512":59,"513":48,"514":49,"515":57,"516":49,"517":48,"518":53,"519":68,"520":53,"521":53,"522":70,"523":85,"524":71,"525":83,"526":75,"527":67,"528":54,"529":54,"530":48,"531":57,"532":61,"533":65,"534":43,"535":47,"536":46,"537":46,"538":56,"539":73,"540":71,"541":41,"542":44,"543":38,"544":54,"545":59,"546":48,"547":66,"548":46,"549":53}

page_21_json = {"0":74,"1":74,"2":76,"3":86,"4":90,"5":81,"6":78,"7":79,"8":80,"9":86,"10":90,"11":100,"12":90,"13":80,"14":80,"15":79,"16":78,"17":89,"18":94,"19":79,"20":75,"21":76,"22":74,"23":80,"24":86,"25":94,"26":81,"27":75,"28":76,"29":78,"30":80,"31":90,"32":93,"33":81,"34":79,"35":76,"36":78,"37":81,"38":91,"39":96,"40":80,"41":78,"42":77,"43":78,"44":77,"45":87,"46":94,"47":81,"48":76,"49":79,"50":81,"51":80,"52":86,"53":92,"54":80,"55":77,"56":77,"57":78,"58":80,"59":86,"60":93,"61":82,"62":78,"63":75,"64":79,"65":83,"66":92,"67":98,"68":80,"69":79,"70":79,"71":87,"72":84,"73":93,"74":95,"75":85,"76":81,"77":78,"78":82,"79":80,"80":94,"81":96,"82":78,"83":77,"84":75,"85":80,"86":77,"87":88,"88":94,"89":80,"90":81,"91":79,"92":77,"93":81,"94":89,"95":95,"96":83,"97":82,"98":79,"99":80,"100":79,"101":94,"102":95,"103":82,"104":82,"105":79,"106":80,"107":80,"108":89,"109":96,"110":85,"111":85,"112":85,"113":85,"114":87,"115":97,"116":95,"117":86,"118":86,"119":83,"120":78,"121":81,"122":92,"123":96,"124":84,"125":82,"126":80,"127":82,"128":81,"129":93,"130":100,"131":83,"132":82,"133":86,"134":84,"135":86,"136":92,"137":98,"138":85,"139":84,"140":85,"141":81,"142":85,"143":93,"144":95,"145":83,"146":80,"147":78,"148":79,"149":82,"150":93,"151":94,"152":87,"153":81,"154":82,"155":82,"156":83,"157":88,"158":98,"159":84,"160":79,"161":79,"162":81,"163":81,"164":91,"165":99,"166":82,"167":79,"168":77,"169":80,"170":81,"171":91,"172":96,"173":83,"174":82,"175":81,"176":78,"177":80,"178":95,"179":97,"180":86,"181":81,"182":82,"183":79,"184":82,"185":89,"186":91,"187":80,"188":78,"189":74,"190":76,"191":79,"192":89,"193":93,"194":82,"195":76,"196":76,"197":76,"198":80,"199":88,"200":96,"201":84,"202":77,"203":76,"204":78,"205":78,"206":88,"207":92,"208":81,"209":81,"210":81,"211":78,"212":78,"213":91,"214":95,"215":82,"216":78,"217":79,"218":80,"219":78,"220":90,"221":93,"222":78,"223":76,"224":76,"225":78,"226":81,"227":89,"228":97,"229":81,"230":82,"231":82,"232":79,"233":81,"234":94,"235":97,"236":80,"237":76,"238":66,"239":73,"240":81,"241":94,"242":96,"243":83,"244":80,"245":78,"246":77,"247":79,"248":100,"249":100,"250":77,"251":74,"252":76,"253":76,"254":76,"255":87,"256":89,"257":74,"258":75,"259":73,"260":77,"261":79,"262":85,"263":92,"264":77,"265":74,"266":75,"267":78,"268":75,"269":84,"270":92,"271":78,"272":73,"273":73,"274":75,"275":73,"276":86,"277":90,"278":80,"279":80,"280":78,"281":82,"282":79,"283":83,"284":87,"285":89,"286":87,"287":92,"288":91,"289":90,"290":82,"291":74,"292":74,"293":78,"294":85,"295":91,"296":78,"297":76,"298":77,"299":76,"300":76,"301":86,"302":90,"303":78,"304":76,"305":76,"306":75,"307":80,"308":90,"309":100,"310":84,"311":80,"312":78,"313":75,"314":77,"315":88,"316":93,"317":80,"318":75,"319":74,"320":74,"321":76,"322":82,"323":91,"324":76,"325":76,"326":75,"327":75,"328":75,"329":88,"330":91,"331":78,"332":75,"333":75,"334":75,"335":78,"336":86,"337":93,"338":78,"339":74,"340":76,"341":76,"342":73,"343":86,"344":89,"345":74,"346":75,"347":76,"348":74,"349":72,"350":83,"351":87,"352":77,"353":76,"354":75,"355":73,"356":77,"357":86,"358":89,"359":76,"360":74,"361":69,"362":71,"363":71,"364":83,"365":87,"366":73,"367":72,"368":71,"369":72,"370":73,"371":86,"372":84,"373":73,"374":74,"375":70,"376":72,"377":75,"378":84,"379":88,"380":73,"381":70,"382":69,"383":71,"384":71,"385":82,"386":84,"387":71,"388":72,"389":71,"390":70,"391":72,"392":82,"393":89,"394":77,"395":71,"396":71,"397":69,"398":72,"399":84,"400":85,"401":73,"402":69,"403":71,"404":71,"405":73,"406":84,"407":85,"408":71,"409":73,"410":71,"411":73,"412":74,"413":81,"414":87,"415":76,"416":72,"417":73,"418":71,"419":72,"420":85,"421":86,"422":73,"423":71,"424":72,"425":71,"426":71,"427":79,"428":87,"429":70,"430":71,"431":70,"432":72,"433":73,"434":81,"435":89,"436":70,"437":71,"438":68,"439":71,"440":72,"441":84,"442":86,"443":73,"444":71,"445":70,"446":70,"447":72,"448":83,"449":87,"450":76,"451":75,"452":72,"453":72,"454":74,"455":85,"456":88,"457":75,"458":74,"459":71,"460":74,"461":76,"462":87,"463":92,"464":81,"465":82,"466":82,"467":80,"468":82,"469":97,"470":99,"471":91,"472":86,"473":89,"474":79,"475":83,"476":81,"477":84,"478":70,"479":66,"480":65,"481":64,"482":64,"483":74,"484":76,"485":64,"486":58,"487":61,"488":58,"489":59,"490":71,"491":77,"492":64,"493":59,"494":60,"495":58,"496":64,"497":71,"498":73,"499":64,"500":62,"501":61,"502":60,"503":62,"504":68,"505":73,"506":63,"507":59,"508":59,"509":63,"510":62,"511":70,"512":75,"513":64,"514":63,"515":61,"516":60,"517":62,"518":69,"519":72,"520":63,"521":62,"522":60,"523":61,"524":60,"525":73,"526":73,"527":61,"528":60,"529":58,"530":59,"531":60,"532":69,"533":71,"534":58,"535":59,"536":59,"537":60,"538":58,"539":67,"540":70,"541":61,"542":59,"543":58,"544":58,"545":58,"546":69,"547":71,"548":61,"549":60}
lists = sorted(page_20_json.items()) # sorted by key, return a list of tuples

x, y20 = zip(*lists) # unpack a list of pairs into two tuples



lists = sorted(page_21_json.items()) # sorted by key, return a list of tuples

x, y21 = zip(*lists) # unpack a list of pairs into two tuples



plot_views(20)

plot_trend(20, y20)

plot_views(21)

plot_trend(21, y21)
# Run

# git clone https://github.com/MohmadAyman/web-traffic-forecast

# cd dl_trends/



# modify start and end.



import subprocess



start = 0

end = 27   

ks = [i for i in range(start, end)]



for i in ks:

    bashCommand = "node index.js " + str(i)

    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)