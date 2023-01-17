#from scrappy import Selector
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors

l1=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=action&sort=num_votes,desc")
c1=l1.content
soup=BeautifulSoup(c1,"html.parser")
l2=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=drama&sort=num_votes,desc")
c2=l2.content
soup2=BeautifulSoup(c2,"html.parser")
l3=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=romance&sort=num_votes,desc")
c3=l3.content
soup3=BeautifulSoup(c3,"html.parser")
l4=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=sci-fi&sort=num_votes,desc")
c4=l4.content
soup4=BeautifulSoup(c4,"html.parser")
l5=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=comedy&sort=num_votes,desc")
c5=l5.content
soup5=BeautifulSoup(c5,"html.parser")
l6=requests.get("https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=animation&sort=num_votes,desc")
c6=l6.content
soup6=BeautifulSoup(c6,"html.parser")


movies=[]
#Scraping for movie names
for h3 in soup.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist=pd.DataFrame(movies,columns=["MovieTitle"])
#print(movielist)
Act=["Action"]*50
#print(Action)
movielist["Genre"]=pd.Series(Act)
#print(movielist)

#SCRAPING movie ratings
rate=[]
for div in soup.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate.append(r.text)
#print(len(rate))

movielist["Rating"]=pd.Series(rate)
#print(movielist)

#SCRAPING Number of Votes
Vote=[]
for p in soup.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes=[]
for i in range(0,100,2):
    V=Vote[i]
    Votes.append(V)
#print(Votes)
Gr=[]
for j in range(1,100,2):
    G=Vote[j]
    Gr.append(G)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist["Votercount"]=pd.Series(Votes)
movielist["Gross"]=pd.Series(Gr)
#print(movielist)


#DRAMA MOVIES

movies2=[]
#Scraping for movie names
for h3 in soup2.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies2.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist2=pd.DataFrame(movies2,columns=["MovieTitle"])
#print(movielist)
Dr=["Drama"]*50
#print(Action)
movielist2["Genre"]=pd.Series(Dr)
#print(movielist)

#SCRAPING movie ratings
rate2=[]
for div in soup2.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate2.append(r.text)
#print(len(rate))

movielist2["Rating"]=pd.Series(rate2)
#print(movielist)

#SCRAPING Number of Votes
Vote2=[]
for p in soup2.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote2.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes2=[]
for i in range(0,100,2):
    V=Vote2[i]
    Votes2.append(V)
#print(Votes)
Gr2=[]
for j in range(1,100,2):
    G2=Vote[j]
    Gr2.append(G2)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist2["Votercount"]=pd.Series(Votes2)
movielist2["Gross"]=pd.Series(Gr2)
#print(movielist2)
movielist=movielist.append(movielist2,ignore_index=True)
#print(movielist)

#ROMANCE MOVIES
movies3=[]
#Scraping for movie names
for h3 in soup3.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies3.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist3=pd.DataFrame(movies3,columns=["MovieTitle"])
#print(movielist)
Rom=["Romance"]*50
#print(Action)
movielist3["Genre"]=pd.Series(Rom)
#print(movielist)

#SCRAPING movie ratings
rate3=[]
for div in soup3.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate3.append(r.text)
#print(len(rate))

movielist3["Rating"]=pd.Series(rate3)
#print(movielist)

#SCRAPING Number of Votes
Vote3=[]
for p in soup3.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote3.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes3=[]
for i in range(0,100,2):
    V=Vote3[i]
    Votes3.append(V)
#print(Votes)
Gr3=[]
for j in range(1,100,2):
    G3=Vote[j]
    Gr3.append(G3)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist3["Votercount"]=pd.Series(Votes3)
movielist3["Gross"]=pd.Series(Gr3)
#print(movielist2)
movielist=movielist.append(movielist3,ignore_index=True)


#SCI-FI MOVIE list

movies4=[]
#Scraping for movie names
for h3 in soup4.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies4.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist4=pd.DataFrame(movies4,columns=["MovieTitle"])
#print(movielist)
Scifi=["Sci-Fi"]*50
#print(Action)
movielist4["Genre"]=pd.Series(Scifi)
#print(movielist)

#SCRAPING movie ratings
rate4=[]
for div in soup4.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate4.append(r.text)
#print(len(rate))

movielist4["Rating"]=pd.Series(rate4)
#print(movielist)

#SCRAPING Number of Votes
Vote4=[]
for p in soup4.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote4.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes4=[]
for i in range(0,100,2):
    V=Vote4[i]
    Votes4.append(V)
#print(Votes)
Gr4=[]
for j in range(1,100,2):
    G4=Vote[j]
    Gr4.append(G4)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist4["Votercount"]=pd.Series(Votes4)
movielist4["Gross"]=pd.Series(Gr4)
#print(movielist2)
movielist=movielist.append(movielist4,ignore_index=True)


#COMEDY MOVIES list


movies5=[]
#Scraping for movie names
for h3 in soup5.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies5.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist5=pd.DataFrame(movies5,columns=["MovieTitle"])
#print(movielist)
Com=["Comedy"]*50
#print(Action)
movielist5["Genre"]=pd.Series(Com)
#print(movielist)

#SCRAPING movie ratings
rate5=[]
for div in soup5.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate5.append(r.text)
#print(len(rate))

movielist5["Rating"]=pd.Series(rate5)
#print(movielist)

#SCRAPING Number of Votes
Vote5=[]
for p in soup5.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote5.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes5=[]
for i in range(0,100,2):
    V=Vote5[i]
    Votes5.append(V)
#print(Votes)
Gr5=[]
for j in range(1,100,2):
    G5=Vote[j]
    Gr5.append(G5)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist5["Votercount"]=pd.Series(Votes5)
movielist5["Gross"]=pd.Series(Gr5)
#print(movielist2)
movielist=movielist.append(movielist5,ignore_index=True)



#CREATING ANIMATION list



movies6=[]
#Scraping for movie names
for h3 in soup6.findAll('h3', attrs={'class':'lister-item-header'}):
    #print ("\n")
    for a in h3.findAll('a'):
            movies6.append(a.text)
            
#print(movies)
#CREATING DF with Movie Details
movielist6=pd.DataFrame(movies6,columns=["MovieTitle"])
#print(movielist)
Ani=["Animation"]*50
#print(Action)
movielist6["Genre"]=pd.Series(Ani)
#print(movielist)

#SCRAPING movie ratings
rate6=[]
for div in soup6.findAll('div', attrs={'class':'inline-block ratings-imdb-rating'}):
    #print ("\n")
    for r in div.findAll('strong'):
            rate6.append(r.text)
#print(len(rate))

movielist6["Rating"]=pd.Series(rate6)
#print(movielist)

#SCRAPING Number of Votes
Vote6=[]
for p in soup6.findAll('span', attrs={'name':'nv'}):
    #print ("\n")
    Vote6.append(p.text)
    
    
#SCRAPING to add VoterCount and Gross
Votes6=[]
for i in range(0,100,2):
    V=Vote6[i]
    Votes6.append(V)
#print(Votes)
Gr6=[]
for j in range(1,100,2):
    G6=Vote[j]
    Gr6.append(G6)
#print(Gr)
#print(len(Votes))
#print(len(Gr))
movielist6["Votercount"]=pd.Series(Votes6)
movielist6["Gross"]=pd.Series(Gr6)
#print(movielist2)
movielist=movielist.append(movielist6,ignore_index=True)



'''x=[]
for z in soup6.findAll('div',attrs={'class':'lister-item-content'}):
    for p in z.findAll('p'):
        w=p.text
        s1=w.find("Director")
        s2=w.find("|")
        d=(w[s1:s2])
        x.append(d)
print(x)'''
movielist


#Fixing the movielist dataframe column GrossProtift and removing duplicate movies 


movielist["GrossI"]=movielist["Gross"].str.split("$")
movielist["G1"]=movielist.GrossI.str.get(1)
del movielist["GrossI"]
movielist["G2"]=movielist["G1"].str.split("M")
movielist["GrossProfit"]=movielist["G2"].str.get(0)
del movielist["G2"]
del movielist["G1"]
del movielist["Gross"]
movielist["GrossProfit"]=pd.to_numeric(movielist["GrossProfit"])
movielist["Rating"]=pd.to_numeric(movielist["Rating"])
Genres=movielist["Genre"].unique()
movielist = movielist.drop_duplicates(subset='MovieTitle', keep="first")
print(Genres)

#GENREWISE summation of Gross profits
#PLOTTING GRAPH for Gross prod per Genre

Gross=movielist.groupby('Genre')["GrossProfit"].mean()
print(Gross)
fr1={"GrossProfit":Gross}
GenreGross=pd.DataFrame(fr1)

GenreGross.describe()

def grosscol(col):
    if col>315:
        return("indianred")
    elif (col>=305 and col<=315):
        return("tomato")
    else:
        return("gold")



fig,ax=plt.subplots()
fig.set_size_inches(10,8, forward=True)
for ind,gprof in zip(GenreGross.index,GenreGross["GrossProfit"]):
    ax.bar(ind,gprof,color=grosscol(gprof))

    
#Creating Legend:
HighG = mpatches.Patch(color='indianred', label='Gross:>$315M')
MedG = mpatches.Patch(color='tomato', label='Gross:$305M-315M')
LowG = mpatches.Patch(color='gold', label='Gross:<$305M')

ax.set_xlabel("Genre")
ax.set_ylabel("Average Gross in USD(Millions)")
plt.title("Aaverage Gross per movie from various genres",color="indianred",fontsize=18)
plt.legend(handles=[HighG,MedG,LowG],prop={"size":8})
plt.ylim((0,400))
plt.savefig("GenrewiseGrossIncome.png")
plt.show()


#PLOT GENRE WISE TOP 10
#TOP 10 movies from each genre and their ratings
#PLOT ACTION

#####TRY


def grosscol1(val):
    if val>8.5:
        return("mediumvioletred")
    elif (val>=8 and val<=8.5):
        return("deeppink")
    elif (val>=7 and val<=7.9):
        return("pink")
    else:
        return("darkmagenta")

HighR = mpatches.Patch(color='mediumvioletred', label='Rating:>8.5')
MedR = mpatches.Patch(color='deeppink', label='Rating:8-8.5')
LowR = mpatches.Patch(color='pink', label='Rating:7-7.9')
BlowR=mpatches.Patch(color='darkmagenta', label='Rating:<7')
#plt.legend(handles=[HighR,MedR,LowR],prop={"size":8})
    
fig,ax=plt.subplots(3,2,figsize=(16,16))
g1=movielist.groupby("Genre")
g11=g1.get_group("Action")
ActTop10=g11.sort_values(["Rating"],ascending=(False)).head(10)
ActTop10=ActTop10.sort_values(["Rating"],ascending=True)
#print(ActTop10)
ActTop10.reset_index(inplace = True) 

##START

fig.tight_layout(pad=2.0)
#ax[0,0].scatter(ActTop10["Rating"],ActTop10["MovieTitle"],marker="h",color='green',s=150,edgecolors="black",label="ACTION")
#ax[0,0].barh(ActTop10["MovieTitle"],ActTop10["Rating"],color=grosscol1(ActTop10))

for mov,rate in zip(ActTop10["MovieTitle"],ActTop10["Rating"]):
    ax[0,0].barh(mov,rate,color=grosscol1(rate))

ax[0,0].set_xlabel("Rating")
#ax[0,0].set_ylabel("Rating")
ax[0,0].set_title("IMDb Ratings:Top 10 Action Movies")



#PLOT ANIMATION
fig.tight_layout(pad=2.0)
g2=movielist.groupby("Genre")
g22=g2.get_group("Animation")
AniTop10=g22.sort_values(["Rating"],ascending=(False)).head(10)

AniTop10=AniTop10.sort_values(["Rating"],ascending=True)
AniTop10.reset_index(inplace = True) 
#rint(AniTop10)


#ax[0,1].barh(AniTop10["MovieTitle"],AniTop10["Rating"],color='yellow')

for mov,rate in zip(AniTop10["MovieTitle"],AniTop10["Rating"]):
    ax[0,1].barh(mov,rate,color=grosscol1(rate))
#ax[0,1].set_xticklabels(AniTop10["MovieTitle"],rotation=90)
ax[0,1].set_xlabel("Rating")

ax[0,1].set_title("IMDb Ratings:Top 10 Animation Movies")
#ax[0,1].set_ylim([0,10])


#PLOT COMEDY

fig.tight_layout(pad=2.0)
g3=movielist.groupby("Genre")
g33=g3.get_group("Comedy")
ComTop10=g33.sort_values(["Rating"],ascending=(False)).head(10)
ComTop10=ComTop10.sort_values(["Rating"],ascending=True)
ComTop10.reset_index(inplace = True) 
#rint(AniTop10)

#ax[1,0].barh(ComTop10["MovieTitle"],ComTop10["Rating"],color='cyan')

for mov,rate in zip(ComTop10["MovieTitle"],ComTop10["Rating"]):
    ax[1,0].barh(mov,rate,color=grosscol1(rate))

ax[1,0].set_xlabel("Rating")

ax[1,0].set_title("IMDb Ratings:Top 10 Comedy Movies")


#PLOT DRAMA

fig.tight_layout(pad=2.0)
g4=movielist.groupby("Genre")
g44=g4.get_group("Drama")
DraTop10=g44.sort_values(["Rating"],ascending=(False)).head(10)
DraTop10=DraTop10.sort_values(["Rating"],ascending=True)
DraTop10.reset_index(inplace = True) 
#rint(AniTop10)

#ax[1,1].barh(DraTop10["MovieTitle"],DraTop10["Rating"],color='pink')
#ax[1,1].set_xticklabels(DraTop10["MovieTitle"],rotation=90)

for mov,rate in zip(DraTop10["MovieTitle"],DraTop10["Rating"]):
    ax[1,1].barh(mov,rate,color=grosscol1(rate))
ax[1,1].set_xlabel("Rating")

ax[1,1].set_title("IMDb Ratings:Top 10 Drama Movies")


#ROMANCE PLOT

fig.tight_layout(pad=2.0)
g5=movielist.groupby("Genre")
g55=g5.get_group("Romance")
RomTop10=g55.sort_values(["Rating"],ascending=(False)).head(10)
RomTop10=RomTop10.sort_values(["Rating"],ascending=True)
RomTop10.reset_index(inplace = True) 
#rint(AniTop10)

#ax[2,0].barh(RomTop10["MovieTitle"],RomTop10["Rating"],color='purple')

for mov,rate in zip(RomTop10["MovieTitle"],RomTop10["Rating"]):
    ax[2,0].barh(mov,rate,color=grosscol1(rate))

ax[2,0].set_xlabel("Rating")

ax[2,0].set_title("IMDb Ratings:Top 10 Romance Movies")


#SCIFI PLOT

fig.tight_layout(pad=2.0)
g6=movielist.groupby("Genre")
g66=g6.get_group("Sci-Fi")
SciTop10=g66.sort_values(["Rating"],ascending=(False)).head(10)
SciTop10=SciTop10.sort_values(["Rating"],ascending=True)
SciTop10.reset_index(inplace = True) 
#rint(AniTop10)

#ax[2,1].barh(SciTop10["MovieTitle"],SciTop10["Rating"],color='red')
for mov,rate in zip(SciTop10["MovieTitle"],SciTop10["Rating"]):
    ax[2,1].barh(mov,rate,color=grosscol1(rate))

ax[2,1].set_xlabel("Rating")

ax[2,1].set_title("IMDb Ratings:Top 10 Sci-Fi Movies")


plt.suptitle("IMDb Ratings for Top 10 Movies in Each Genre",fontsize=34,color="indianred",y=1.)

#Displaying the legend
fig.tight_layout(rect=(0,0,1,0.9))
fig.legend(handles=[HighR,MedR,LowR,BlowR],prop={"size":14},loc="upper right")
plt.savefig('IMDb Genre Ratings', ext='png', bbox_inches="tight") 
plt.show()

movielist

#AVERAGE RATING PER GENRE
GenAvgR=movielist.groupby("Genre")["Rating"].mean()

#LEGEND
def grosscol2(val):
    if val>8.2:
        return("darkred")
    elif (val>7.8 and val<=8.2):
        return("orangered")
    else:
        return("orange")

HighRt = mpatches.Patch(color='darkred', label='Rating:>8.2')
MedRt = mpatches.Patch(color='orangered', label='Rating:7.8-8.2')
LowRt = mpatches.Patch(color='orange', label='Rating:<7.8')



fig,ax=plt.subplots()
fig.set_size_inches(10,8, forward=True)
for gen,avrate in zip(Genres,GenAvgR):
    ax.bar(gen,avrate,color=grosscol2(avrate))
#ax.bar(Genres,GenAvgR)
ax.set_xticklabels(Genres,rotation=90)
ax.set_ylim(0,10)
ax.set_ylabel("Average IMDb Rating",fontsize=12)
plt.title("Average IMDb Rating for Top 219 Movies from various genres",fontsize=20,color="indianred")
ax.legend(handles=[HighRt,MedRt,LowRt],prop={"size":12},loc="upper right")
#fig.tight_layout(rect=(0,0,1,0.9))
plt.savefig("Genrewise Average IMDb Rating.png")
plt.show()


#Most common IMDb Ratings
fig,ax=plt.subplots()
fig.set_size_inches(10,8, forward=True)
N, bins, patches=ax.hist(movielist["Rating"],bins=25,histtype="bar",color="crimson",edgecolor="orange",linewidth=2)
#print(counts)
#print(bins)
#ax.set_xlim(0,10)
ax.set_xlabel("IMDb Rating",fontsize=10)
ax.set_ylabel("Number of Movies",fontsize=10)
ax.set_xticks(bins)
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))


##############

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)


#############
plt.title("Number of Movies per Rating",fontsize=16,color="indianred")
plt.savefig("HistRating.png")
plt.show()
print(movielist)


rr=movielist.groupby('Genre')["MovieTitle"].count()
print(rr)
print(movielist.head(219))