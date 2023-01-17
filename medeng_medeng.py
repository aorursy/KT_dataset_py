from difflib import SequenceMatcher

import re
path='/kaggle/input/similarity-info/'

with open(path+'Human299EGenome.txt', 'r') as f:

    H299EGenome = f.readlines()



H299EGenome = ''.join(H299EGenome)

H299EGenome = H299EGenome.replace('\n','')



H299EGenes = re.findall('.{%d}' % 60, H299EGenome)



open("Human299EGenes2.txt", "w").close()

f = open("Human299EGenes2.txt", "a")



for gene in H299EGenes:

    f.write(gene+'\n')
with open(path+'MiddleEastGenome.txt', 'r') as f:

    MEGenome = f.readlines()





MEGenome = ''.join(MEGenome)

MEGenome = MEGenome.replace('\n','')



MEGenes = re.findall('.{%d}' % 60, MEGenome)



open("MiddleEastGenes2.txt", "w").close()

f = open("MiddleEastGenes2.txt", "a")



for gene in MEGenes:

    f.write(gene+'\n')

with open(path+'SARSGenome.txt', 'r') as f:

    SARSGenome = f.readlines()





SARSGenome = ''.join(SARSGenome)

SARSGenome = SARSGenome.replace('\n','')



SARSGenes = re.findall('.{%d}' % 60, SARSGenome)



open("SARSGenes2.txt", "w").close()

f = open("SARSGenes2.txt", "a")



for gene in SARSGenes:

    f.write(gene+'\n')

open("SARSvs299E.txt", "w").close()

diffSARS = open("SARSvs299E.txt", "a")



open("MEvs299E.txt", "w").close()

diffME = open("MEvs299E.txt", "a")





for index in range(len(H299EGenes)):

    s = SequenceMatcher(lambda x: x == " ", H299EGenes[index], MEGenes[index])

    diffME.write(str((index*60)+1)+"    ")

    diffME.write(str(round(s.ratio(),3)))

    diffME.write('\n')



for index in range(len(H299EGenes)):

    s = SequenceMatcher(lambda x: x == " ", H299EGenes[index], SARSGenes[index])

    diffSARS.write(str((index*60)+1)+"    ")

    diffSARS.write(str(round(s.ratio(),3)))

    diffSARS.write('\n')
import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

def read_data(file_name):

    """parse data file and  the locations and similarity data 

      Params:

       file name:path to file 



        Returns:

         locations:A list  corresponds to the locations of genes of a 

         certain virus



        ratios:A  lists  corresponds to the similarity ratio of genes of a 

        certain virus and the virus under study (covid-19)

      

        """  



    locations=[] ## to store gene location

    ratios=[]    ## to store similarity ratio with covid-19

    with open(file_name,'r') as f:

      lines=f.readlines()

      for line in lines:

        loc=line.split(' ')[0]

        sim=line.split(' ')[4].strip('\n')

        locations.append(int(loc))

        ratios.append(float(sim))

      return locations,ratios
class Plotter():

  def __init__(self,rows,cols,names):

    self.fig = make_subplots(rows=rows+1, cols=cols,subplot_titles=(names))

    self.rows=rows+1 ##no of rows to be seen when visulaizing sublots change to use more or less rows

    self.cols=cols ##no of columns to be seen when visulaizing sublots change to use more or less columns

    self.names=names ##names of virsus to be visulaized

  def plot_line(self,fig,loc_1,ratios_1,virus_name,row,col):

    """Visualize the similarites between viruses,with gene locations on x-axis and similarity 

      ratios on y-axis



        Params:

        locations:A list corresponds to the locations of genes of a 

        certain virus



        ratios:A list corresponds to the similarity ratio of genes of a 

        certain virus and the virus under study (covid-19) 

        """  

  

    fig.add_trace(go.Scatter(x=loc_1, y=ratios_1,

                        mode='lines',

                        name=virus_name),row=row, col=col)

    



  def plot_multiline(self,fig,loc,ratios,names,row,col):

    """Visualize the similarites between viruses,with gene locations on x-axis and similarity 

      ratios on y-axis and plots lines corresponding to all virus on same plot



        Params:

        locations:A list of lists where each list corresponds to the locations of genes of a 

        certain virus



        ratios:A list of lists where each list corresponds to the similarity ratio of genes of a 

        certain virus and the virus under study (covid-19) 

     

       """  



    assert len(locations)==len(ratios), "locations and ratios should be of same length"

    no_virus2visulize=len(locations)

    for v in range(0,no_virus2visulize):

      fig.add_trace(go.Scatter(x=locations[v], y=ratios[v],

                      mode='lines', 

                      name=names[v]),row=row, col=col)

    

  

  def visulaize_lines(self,locations,ratios):

    """Visualize the similarites between viruses,with gene locations on x-axis and similarity 

      ratios on y-axis



        Params:

        locations:A list of lists where each list corresponds to the locations of genes of a 

        certain virus



        ratios:A list of lists where each list corresponds to the similarity ratio of genes of a 

        certain virus and the virus under study (covid-19) 



        names:A list of names of viruses to be visualized

        """  

    assert len(locations)==len(ratios), "locations and ratios should be of same length"

    no_virus2visulize=len(locations)

    r=1

    c=0

    

    for v in range(0,no_virus2visulize):

      if(c<self.cols):

        self.plot_line(self.fig,locations[v],ratios[v],self.names[v],row=r,col=c+1)

        c+=1

      else:

        if(r<self.rows):

          r=r+1

          self.plot_line(self.fig,locations[v],ratios[v],self.names[v],row=r,col=c)

 

    self.plot_multiline(self.fig,locations,ratios,names,r+1,c)

    self.fig.show()

locations_1,ratios_1=read_data(path+'MEvs299E.txt')

locations_2,ratios_2=read_data(path+'SARSvs299E.txt')

rows=2

cols=1

names=['Mers','Sars']

plt= Plotter(rows,cols,names)

locations=[locations_1,locations_2]

ratios=[ratios_1,ratios_2]

plt.visulaize_lines(locations,ratios)
