#source:  https://www.kaggle.com/tanulsingh077/a-comprehensive-resource-notebook-for-beginners

from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('mWeJ0lakG_A',width=600, height=400)
from Bio import SeqIO

record = SeqIO.read("../input/sample.fna", "fasta")
new_seq = str(record.seq[0:200])
!pip install dna_features_viewer
record = GraphicRecord(sequence=new_seq, features=[

    GraphicFeature(start=5, end=10, strand=+1, color='#ffcccc'),

    GraphicFeature(start=8, end=15, strand=+1, color='#ccccff')

])



ax, _ = record.plot(figure_width=5)

record.plot_sequence(ax)

record.plot_translation(ax, (8, 23), fontdict={'weight': 'bold'})

ax.figure.savefig('sequence_and_translation.png', bbox_inches='tight')