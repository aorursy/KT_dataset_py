filename = '../input/sars-coronavirus-accession/SARS_CORONAVIRUS_NC_045512_sequence.fasta'



covid_genome = "" 

with open(filename) as f: 

    genome = f.read() 



covid_genome = genome.replace(">NC_045512.2 Wuhan seafood market pneumonia virus isolate Wuhan-Hu-1, complete genome", "").replace("\n", "")

print(genome)

b = covid_genome.encode()



print("Corona Virus Uncompressed File Size: " + str(len(b) / 1000) + " KB")
transcribed_genome = covid_genome.replace('T', 'U')

covid_codons = [transcribed_genome[i:i+3] for i in range(0, len(transcribed_genome), 3)]

print(covid_codons)    
import pandas as pd

df = pd.read_csv('../input/codons/Codons - Sheet1 (1).csv')



amino_acids = []



for codon in covid_codons: 

    try: 

        index = df.index[df['codon'] == codon][0]

        amino_acids.append(df.at[index, 'letter'])

    except: 

        continue 



protien_sequence = ''.join(amino_acids)

print(protien_sequence)
x = protien_sequence.encode() 

print(str(len(x) / 1000) + ' KB')
unique_acids = [] 

for acid in protien_sequence: 

    if acid not in unique_acids: 

        unique_acids.append(acid)

data = {}

for acid in unique_acids: 

    data[acid] = protien_sequence.count(acid)

df = pd.read_csv('../input/codons/Codons - names.csv')

df.tail()

for letter in data.keys():

    try: 

        index = df.index[df['One Letter Code'] == letter][0]

        val = data[letter]

        new_title = df.at[index, 'name']

        del data[letter]

        data[new_title] = val 

    except: 

        continue 



#remove stop/start codons

del data['K']

del data['O']

    
import matplotlib.pyplot as plt

fig = plt.figure() 

ax = fig.add_axes([0 ,1, 3.5, 2])

ax.set_title('COVID-19 Amino Acids', fontsize=20)

ax.bar(data.keys(), data.values())

plt.show()