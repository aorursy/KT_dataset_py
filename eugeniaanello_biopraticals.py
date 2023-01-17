from Bio.Seq import Seq
# from Bio.Alphabet import generic_dna, generic_protein
from Bio import SeqIO
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo
seq_records = list(SeqIO.parse("../input/biodata/p53_ensemble.fa", "fasta"))
print(seq_records[0])
print(type(seq_records))
print(type(seq_records[0]))
dir(seq_records[0])
print(seq_records[0].seq)
print(seq_records[0].id)
print(seq_records[0].name)
print(seq_records[0].description)
print(seq_records[0].description.split(':'))
data = {}  # {molecule_type: [seq_record, ...]}
for rec in seq_records:
    # print(rec)

    # Change alphabet (only for old Python)
    # if 'peptide' in rec.description:
    #     rec.seq = Seq(str(rec.seq), generic_protein)
    # else:
    #     rec.seq = Seq(str(rec.seq), generic_dna)


    # Parse the sequence type (peptide, cds, cdna, utr5, utr3, <x>_exon, intron_<x>)
    seq_type = "_".join(rec.description.split(':')[0].split()[1:])
    data[seq_type] = rec
    print("############################################################## ")
    print(seq_type)
    print("############################################################## ")
    print(rec)
for index, record_type in enumerate(data):
      print("#### Protein sequence",index+1,":")
      print(record_type, data[record_type], sep="\n\n")
      print()    
print(data["cds"].seq, end="\n\n")
print(data["cds"].seq.transcribe(), end="\n\n") 
print(data["cds"].seq.transcribe().translate(), end="\n\n")
print(data["cds"].seq.translate(), end="\n\n") 
print(data["cds"].seq.complement().transcribe().translate(), end="\n\n")
dir(Seq)
for k in data.keys():
    if "exon" in k:
        exon_position_cds = data["cds"].seq.find(data[k].seq)
        print("{} {:>5} {:>5} {}...".format(k, exon_position_cds, len(data[k].seq), data[k].seq[0:10]))
print()
for k in data.keys():
    if "exon" in k:
        exon_position_cds = data["cds"].seq.find(data[k].seq)
        exon_position_cdna = data["cdna"].seq.find(data[k].seq)
        print("{} {:>5} {:>5} {}... (length {:>5})".format(k, exon_position_cds, exon_position_cdna, data[k].seq[0:10], len(data[k].seq)))
print()
cds_position_cdna = data["cdna"].seq.find(data["cds"].seq)
print("cdna_len:{} cds_len:{}".format(len(data["cdna"].seq), len(data["cds"].seq)))

print("{:>4} {}...\n   0 {}...".format(cds_position_cdna,
                                       data["cdna"].seq[cds_position_cdna:cds_position_cdna + 10],
                                       data["cds"].seq[:10]))

print("...{}... {}\n...{}    {}".format(data["cdna"].seq[cds_position_cdna + len(data["cds"].seq) - 10:cds_position_cdna + len(data["cds"].seq)],
                                        cds_position_cdna + len(data["cds"].seq),
                                        data["cds"].seq[-10:], len(data["cds"].seq)))
print()
dir(pairwise2)
dir(MatrixInfo)
alignments = pairwise2.align.localds(data["cds"].seq, data["cdna"].seq, MatrixInfo.ident, -200, -0.5, one_alignment_only=True)
print(len(alignments))
print(alignments)
#print(pairwise2.format_alignment(*alignments[0]))
alignments = []
for k in sorted(data.keys()):
    if "exon" in k:

        # Test different types of alignment
        # alignment = pairwise2.align.globalxx(data["cds"].seq, data[k].seq, one_alignment_only=True)[0]
        # alignment = pairwise2.align.localxx(data["cds"].seq, data[k].seq, one_alignment_only=True)[0]
        # alignment = pairwise2.align.localxs(data["cds"].seq, data[k].seq, -10, -0.5, one_alignment_only=True)[0]
        alignment = pairwise2.align.localds(data["cds"].seq, data[k].seq, MatrixInfo.ident, -200, -0.5, one_alignment_only=True)[0]

        # Calculate average score per alignment position
        seq1_aligned, seq2_aligned, score, alignment_begin, alignment_end = alignment
        score_norm = score / float(alignment_end - alignment_begin)

        alignments.append((k, score, score_norm, alignment_begin, alignment_end, alignment))
print(len(alignments))
#print(alignments[0])
print(alignments[1])

#score_norm
print(alignments[1][2])
for alignment in sorted(alignments, key=lambda ele: ele[3]):

    print("{} score:{} s_norm:{:.2f} start:{} end:{}".format(*alignment[:5]))
    print(pairwise2.format_alignment(*alignment[-1]))

print()
#let's translate this mRNA into the corresponding protein sequence
print(data['cdna'].seq.translate())
alignments = []
for k in data.keys():
    if "exon" in k:
        for i in range(0, 3):
            exon_peptide = data[k].seq[i:].translate(stop_symbol="")

            alignment = pairwise2.align.localds(data["peptide"].seq, exon_peptide, MatrixInfo.blosum62, -100, -0.5, one_alignment_only=True)[0]

            # Calculate average score per alignment position
            seq1_aligned, seq2_aligned, score, alignment_begin, alignment_end = alignment
            score_norm = score / float(alignment_end - alignment_begin)
            cov_exon = (alignment_end - alignment_begin) / len(exon_peptide)

            alignments.append((k, score, score_norm, alignment_begin, alignment_end, len(exon_peptide), cov_exon, alignment))
for alignment in alignments:
    # Filter out bad alignments, i.e. those with average bit score per position lower than 5. Lower scores
    # indicate there are gaps or unfavored mutations, here we are looking for an exact match.
    if alignment[2] > 5:
        print("{}\n  score:{}\n  score_normalized:{:.2f}\n  peptide_alignment_start:{}\n  peptide_alignment_end:{}\n  exon_len:{}\n  cov_exon:{:.2f}\n".format(*alignment[:-1]))
        print(pairwise2.format_alignment(*alignment[-1]))

print()