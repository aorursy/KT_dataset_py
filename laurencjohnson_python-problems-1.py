input_dna = "ATCGCGAT"



#to replace letters 



mRNA = input_dna.replace('T','U')



print(mRNA)
import Bio



#testing 

from Bio.Seq import Seq

my_seq = Seq("GCCGCAT")

print(my_seq.reverse_complement())



# given sequence

from Bio.Seq import Seq

my_seq = Seq("TGCGCGGATCGTACCTAATCGATGGCATTAGCCGAGCCCGATTACGC")

print(my_seq.reverse_complement())









from Bio.Seq import Seq

my_seq = Seq("GATTCTCTGGAGAGAAGCTTCTCTCCAGAGAATC")



if   str("GATTCTCTGGAGAGAAGCTTCTCTCCAGAGAATC") == str(my_seq.reverse_complement()):

        print("True")

    

else:   print("False")






