import numpy as np
from collections import Counter
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
QUERY = """
        SELECT sample_path,REGEXP_EXTRACT_ALL(content, r"[_a-zA-Z][_a-zA-Z0-9]{0,30}['\s']*=") as line
        FROM `bigquery-public-data.github_repos.sample_contents` where sample_path like '%.py' """
print ("QUERY SIZE:   ")
print (str(round((bq_assistant.estimate_query_size(QUERY)),2))+str(" GB"))
result=bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=25)
def Create_WordCloud(Frequency):
    wordcloud = WordCloud(background_color='black',
                              random_state=42).generate_from_frequencies(Frequency)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):
    list_of_tuples=list_of_tuples[:items_to_show]
    data = [go.Bar(
            x=[val[0] for val in list_of_tuples],
            y=[val[1] for val in list_of_tuples]
    )]
    layout = go.Layout(
    title=title,xaxis=dict(
        autotick=False,
        tickangle=290 ),)
    fig = go.Figure(data=data, layout=layout)
    #py.offline.iplot(data,layout=layout)
    
    py.offline.iplot(fig)
def RemoveFunc(x):
    return [xx.replace("=","").replace(" ","") for xx in x]
#Basic cleaning of data

result["line"]=result.line.apply(RemoveFunc)
result=result[result.astype(str)['line'] != '[]']
result["sample_path"]=result["sample_path"].apply(lambda x: x.split('/')[-1:][0])
all_variables={}
for x in result.line:
    val=list(set(x))
    for v in val:
        if v not in all_variables:
            all_variables[v]=0
        all_variables[v]+=1
print ("_________WORDCLOUD SHOWING THE MOST USED VARIABLE NAMES IN PYTHON______________")
Create_WordCloud(all_variables)
d=Counter(all_variables)
Create_Bar_plotly(d.most_common(40))
# lets find the most common single character variable in python
single_variables=[]
single_dict={}
for x in all_variables.keys():
    if len(x)==1:
        single_variables.append((x,all_variables[x]))
        single_dict[x]=all_variables[x]
Create_Bar_plotly(sorted(single_variables,key=lambda x: x[1], reverse=True), 10)
# lets find the most common single character variable in python
single_variables=[]
for x in all_variables.keys():
    if len(x)==1:
        single_variables.append((x,all_variables[x]))
Create_Bar_plotly(sorted(single_variables,key=lambda x: x[1]), 10)
#now this was the analysis with respect to all the files present.
#lets find out the average number of variables declared in a single python file
Count_Variables={}
for i,x in result.iterrows():
    Count_Variables[x["sample_path"]]=len(set(x["line"]))
d=Counter(Count_Variables)
Create_Bar_plotly(d.most_common(10))
d.most_common(1)
    
summ=0
for x in Count_Variables.keys():
    summ+=Count_Variables[x]
print ("AVERAGE NUMBER OF VARIABLES WHICH ARE THERE IN A PYTHON PROGRAM")
print (int(float(summ)/float(len(Count_Variables))))
#lets find out the CamelCase  vs use of _ in program vs __identifier__
mydic={}
import re
re_of__val__=re.compile(r"__[\d\s\w\W]+__")
re_of_val_=re.compile(r"_[\d\s\w\W]+_")
mydic["_val_"]=0
mydic["__val__"]=0
mydic["With_Number"]=0
mydic["title_case"]=0
mydic["other"]=0
mydic["uppercase"]=0
mydic["lowercase"]=0
for x in all_variables.keys():
    if re_of__val__.match(x):
        mydic["__val__"]+=1
    elif re_of_val_.match(x):
        mydic["_val_"]+=1
    elif len(re.findall(r"[0-9]+",x))!=0:
        mydic["With_Number"]+=1
    elif x.istitle():
        mydic["title_case"]+=1
    elif x.islower():
        mydic["lowercase"]+=1
    elif x.isupper():
        mydic["uppercase"]+=1
    else:
        mydic["other"]+=1
md=Counter(mydic)
Create_Bar_plotly(md.most_common())
BAD_WORDS="4r5e,50 yard cunt punt\xa0\xa0\xa0,5h1t,5hit,a_s_s,a2m,a55,adult,amateur,anal,anal impaler\xa0\xa0\xa0,anal leakage\xa0\xa0\xa0,anilingus,anus,ar5e,arrse,arse,arsehole,ass,ass fuck\xa0\xa0\xa0,asses,assfucker,ass-fucker,assfukka,asshole,asshole,assholes,assmucus\xa0\xa0\xa0,assmunch,asswhole,autoerotic,b!tch,b00bs,b17ch,b1tch,ballbag,ballsack,bang (one's) box\xa0\xa0\xa0,bangbros,bareback,bastard,beastial,beastiality,beef curtain\xa0\xa0\xa0,bellend,bestial,bestiality,bi+ch,biatch,bimbos,birdlock,bitch,bitch tit\xa0\xa0\xa0,bitcher,bitchers,bitches,bitchin,bitching,bloody,blow job,blow me\xa0\xa0\xa0,blow mud\xa0\xa0\xa0,blowjob,blowjobs,blue waffle\xa0\xa0\xa0,blumpkin\xa0\xa0\xa0,boiolas,bollock,bollok,boner,boob,boobs,booobs,boooobs,booooobs,booooooobs,breasts,buceta,bugger,bum,bunny fucker,bust a load\xa0\xa0\xa0,busty,butt,butt fuck\xa0\xa0\xa0,butthole,buttmuch,buttplug,c0ck,c0cksucker,carpet muncher,carpetmuncher,cawk,chink,choade\xa0\xa0\xa0,chota bags\xa0\xa0\xa0,cipa,cl1t,clit,clit licker\xa0\xa0\xa0,clitoris,clits,clitty litter\xa0\xa0\xa0,clusterfuck,cnut,cock,cock pocket\xa0\xa0\xa0,cock snot\xa0\xa0\xa0,cockface,cockhead,cockmunch,cockmuncher,cocks,cocksuck ,cocksucked ,cocksucker,cock-sucker,cocksucking,cocksucks ,cocksuka,cocksukka,cok,cokmuncher,coksucka,coon,cop some wood\xa0\xa0\xa0,cornhole\xa0\xa0\xa0,corp whore\xa0\xa0\xa0,cox,cum,cum chugger\xa0\xa0\xa0,cum dumpster\xa0\xa0\xa0,cum freak\xa0\xa0\xa0,cum guzzler\xa0\xa0\xa0,cumdump\xa0\xa0\xa0,cummer,cumming,cums,cumshot,cunilingus,cunillingus,cunnilingus,cunt,cunt hair\xa0\xa0\xa0,cuntbag\xa0\xa0\xa0,cuntlick ,cuntlicker ,cuntlicking ,cunts,cuntsicle\xa0\xa0\xa0,cunt-struck\xa0\xa0\xa0,cut rope\xa0\xa0\xa0,cyalis,cyberfuc,cyberfuck ,cyberfucked ,cyberfucker,cyberfuckers,cyberfucking ,d1ck,damn,dick,dick hole\xa0\xa0\xa0,dick shy\xa0\xa0\xa0,dickhead,dildo,dildos,dink,dinks,dirsa,dirty Sanchez\xa0\xa0\xa0,dlck,dog-fucker,doggie style,doggiestyle,doggin,dogging,donkeyribber,doosh,duche,dyke,eat a dick\xa0\xa0\xa0,eat hair pie\xa0\xa0\xa0,ejaculate,ejaculated,ejaculates ,ejaculating ,ejaculatings,ejaculation,ejakulate,erotic,f u c k,f u c k e r,f_u_c_k,f4nny,facial\xa0\xa0\xa0,fag,fagging,faggitt,faggot,faggs,fagot,fagots,fags,fanny,fannyflaps,fannyfucker,fanyy,fatass,fcuk,fcuker,fcuking,feck,fecker,felching,fellate,fellatio,fingerfuck ,fingerfucked ,fingerfucker ,fingerfuckers,fingerfucking ,fingerfucks ,fist fuck\xa0\xa0\xa0,fistfuck,fistfucked ,fistfucker ,fistfuckers ,fistfucking ,fistfuckings ,fistfucks ,flange,flog the log\xa0\xa0\xa0,fook,fooker,fuck hole\xa0\xa0\xa0,fuck puppet\xa0\xa0\xa0,fuck trophy\xa0\xa0\xa0,fuck yo mama\xa0\xa0\xa0,fuck\xa0\xa0\xa0,fucka,fuck-ass\xa0\xa0\xa0,fuck-bitch\xa0\xa0\xa0,fucked,fucker,fuckers,fuckhead,fuckheads,fuckin,fucking,fuckings,fuckingshitmotherfucker,fuckme ,fuckmeat\xa0\xa0\xa0,fucks,fucktoy\xa0\xa0\xa0,fuckwhit,fuckwit,fudge packer,fudgepacker,fuk,fuker,fukker,fukkin,fuks,fukwhit,fukwit,fux,fux0r,gangbang,gangbang\xa0\xa0\xa0,gang-bang\xa0\xa0\xa0,gangbanged ,gangbangs ,gassy ass\xa0\xa0\xa0,gaylord,gaysex,goatse,god,god damn,god-dam,goddamn,goddamned,god-damned,ham flap\xa0\xa0\xa0,hardcoresex ,hell,heshe,hoar,hoare,hoer,homo,homoerotic,hore,horniest,horny,hotsex,how to kill,how to murdep,jackoff,jack-off ,jap,jerk,jerk-off ,jism,jiz ,jizm ,jizz,kawk,kinky Jesus\xa0\xa0\xa0,knob,knob end,knobead,knobed,knobend,knobend,knobhead,knobjocky,knobjokey,kock,kondum,kondums,kum,kummer,kumming,kums,kunilingus,kwif\xa0\xa0\xa0,l3i+ch,l3itch,labia,LEN,lmao,lmfao,lmfao,lust,lusting,m0f0,m0fo,m45terbate,ma5terb8,ma5terbate,mafugly\xa0\xa0\xa0,masochist,masterb8,masterbat*,masterbat3,masterbate,master-bate,masterbation,masterbations,masturbate,mof0,mofo,mo-fo,mothafuck,mothafucka,mothafuckas,mothafuckaz,mothafucked ,mothafucker,mothafuckers,mothafuckin,mothafucking ,mothafuckings,mothafucks,mother fucker,mother fucker\xa0\xa0\xa0,motherfuck,motherfucked,motherfucker,motherfuckers,motherfuckin,motherfucking,motherfuckings,motherfuckka,motherfucks,muff,muff puff\xa0\xa0\xa0,mutha,muthafecker,muthafuckker,muther,mutherfucker,n1gga,n1gger,nazi,need the dick\xa0\xa0\xa0,nigg3r,nigg4h,nigga,niggah,niggas,niggaz,nigger,niggers ,nob,nob jokey,nobhead,nobjocky,nobjokey,numbnuts,nut butter\xa0\xa0\xa0,nutsack,omg,orgasim ,orgasims ,orgasm,orgasms ,p0rn,pawn,pecker,penis,penisfucker,phonesex,phuck,phuk,phuked,phuking,phukked,phukking,phuks,phuq,pigfucker,pimpis,piss,pissed,pisser,pissers,pisses ,pissflaps,pissin ,pissing,pissoff ,poop,porn,porno,pornography,pornos,prick,pricks ,pron,pube,pusse,pussi,pussies,pussy,pussy fart\xa0\xa0\xa0,pussy palace\xa0\xa0\xa0,pussys ,queaf\xa0\xa0\xa0,queer,rectum,retard,rimjaw,rimming,s hit,s.o.b.,s_h_i_t,sadism,sadist,sandbar\xa0\xa0\xa0,sausage queen\xa0\xa0\xa0,schlong,screwing,scroat,scrote,scrotum,semen,sex,sh!+,sh!t,sh1t,shag,shagger,shaggin,shagging,shemale,shi+,shit,shit fucker\xa0\xa0\xa0,shitdick,shite,shited,shitey,shitfuck,shitfull,shithead,shiting,shitings,shits,shitted,shitter,shitters ,shitting,shittings,shitty ,skank,slope\xa0\xa0\xa0,slut,slut bucket\xa0\xa0\xa0,sluts,smegma,smut,snatch,son-of-a-bitch,spac,spunk,t1tt1e5,t1tties,teets,teez,testical,testicle,tit,tit wank\xa0\xa0\xa0,titfuck,tits,titt,tittie5,tittiefucker,titties,tittyfuck,tittywank,titwank,tosser,turd,tw4t,twat,twathead,twatty,twunt,twunter,v14gra,v1gra,vagina,viagra,vulva,w00se,wang,wank,wanker,wanky,whoar,whore,willies,willy,wtf,xrated,xxx"
BAD_WORDS.lower()
BAD_WORDS=BAD_WORDS.split(",")
a=["god","len"]
BAD_WORDS=list(set(BAD_WORDS).difference(set(a)))
abusive_words={}
for x in all_variables.keys():
    if x in BAD_WORDS:
        abusive_words[x]=int(all_variables[x])
ab=Counter(abusive_words)
Create_Bar_plotly(ab.most_common(10))
QUERY = """
        SELECT sample_path,REGEXP_EXTRACT_ALL(content, r"[_a-zA-Z][_a-zA-Z0-9]{0,30}['\s']*=['\s']*{") as line
        FROM `bigquery-public-data.github_repos.sample_contents` where sample_path like '%.py' """
print ("QUERY SIZE:   ")
print (str(round((bq_assistant.estimate_query_size(QUERY)),2))+str(" GB"))
result=bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=25)
def RemoveFunc(x):
    newlist=[]
    for xx in x:
        val=""
        if xx!="" and xx!=" ":
            Equal=xx.find("=")
            if Equal!=-1:
                val=xx[:Equal]
                val=val.replace(" ","")
                newlist.append(val)
    return newlist
#Basic cleaning of data
result["line"]=result.line.apply(RemoveFunc)
result=result[result.astype(str)['line'] != '[]']
result["sample_path"]=result["sample_path"].apply(lambda x: x.split('/')[-1:][0])
all_variables={}
for x in result.line:
    val=list(set(x))
    for v in val:
        if v not in all_variables:
            all_variables[v]=0
        all_variables[v]+=1
Create_WordCloud(all_variables)
d=Counter(all_variables)
Create_Bar_plotly(d.most_common(20), title="DICTIONARY_NAMES")

Count_Variables={}
for i,x in result.iterrows():
    Count_Variables[x["sample_path"]]=len(set(x["line"]))
d=Counter(Count_Variables)
Create_Bar_plotly(d.most_common(10),title="FILES WITH MOST NUMBER OF DICTIONARY DECLARED")
d.most_common(2)
summ=0
for x in Count_Variables.keys():
    summ+=Count_Variables[x]
print ("AVERAGE NUMBER OF DICTIONARIES WHICH ARE THERE IN A PYTHON PROGRAM")
print (int(float(summ)/float(len(Count_Variables))))
import re
mydic={}
re_of__val__=re.compile(r"__[\d\s\w\W]+__")
re_of_val_=re.compile(r"_[\d\s\w\W]+_")
mydic["_val_"]=0
mydic["__val__"]=0
mydic["With_Number"]=0
mydic["title_case"]=0
mydic["other"]=0
mydic["uppercase"]=0
mydic["lowercase"]=0
for x in all_variables.keys():
    if re_of__val__.match(x):
        mydic["__val__"]+=1
    elif re_of_val_.match(x):
        mydic["_val_"]+=1
    elif len(re.findall(r"[0-9]+",x))!=0:
        mydic["With_Number"]+=1
    elif x.istitle():
        mydic["title_case"]+=1
    elif x.islower():
        mydic["lowercase"]+=1
    elif x.isupper():
        mydic["uppercase"]+=1
    else:
        mydic["other"]+=1
md=Counter(mydic)
Create_Bar_plotly(md.most_common(),title="NAMING CONVENTION FOR NAMING A DICTIONARY")