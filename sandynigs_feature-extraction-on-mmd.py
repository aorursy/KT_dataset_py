#All imports here.

import array

import time

from multiprocessing import Process

import pandas as pd

import numpy as np

import os

from tqdm import tqdm

from math import log

from tqdm import tqdm
#Byte file count vectors.



"""

byte_count_vectors = open("./byte_count_vectors.csv", "w+") # w+ idicates it will create a file if it does not exist in library.



char_list = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']

final_string = "Id,"

for i in char_list:

    for j in char_list:

        concat = ""

        concat = concat.join((i,j))

        final_string = final_string + concat + ","



final_string = final_string+"??"

byte_count_vectors.write(final_string)

byte_count_vectors.write("\n")



files = os.listdir("./byte_files") #byte_files is folder in the local machine having all byte files.

feature_matrix = np.zeros((len(files), 257), dtype = int)



k=0 #Denotes each row. Rows will be total datapoints,here files.

print("The code is still running--------------------------------------------------------------------->>>>>>>>>>>")

for file in tqdm(files):

    f_name = file.split(".")[0]

    byte_count_vectors.write(f_name+",") #This goes into byte_fe_results.csv file.

    with open("./byte_files/"+file, "r") as each_file:

        for line in each_file:

            line = line.rstrip().split(" ") #At the end of each line there is a new space.

            line = line[1:] #Ignored the addresses.

            for hex_word in line:

                if hex_word == "??":

                    feature_matrix[k][256] += 1

                else:

                     feature_matrix[k][int(hex_word, 16)] += 1 #int(hex_word, 16) will return decimal equivalent.

    string_for_each_file = ""

    for i in feature_matrix[k]:

        string_for_each_file = string_for_each_file + str(i) + ","

    string_for_each_file = string_for_each_file[:-1]

    byte_count_vectors.write(string_for_each_file)

    byte_count_vectors.write("\n")

    each_file.close()

    k += 1



byte_count_vectors.close()

print("The code has executed----------------------------------------------------------------------------->>>>>>>>>>>>")



"""
#Byte file size.



"""



def calculate_byte_file_size(class_labels):

#It calculate size of each file in MB.



 fname = [] #Stores the name of each file.

 fsize = [] #Stores the size of each file respectively.

 flabels = [] #Stores respective labels of malware



 for file in tqdm(os.listdir("./byte_files")):

  file_name = file.split('.')[0]

  fname.append(file_name)

  size_in_mb = (os.stat("./byte_files/"+file).st_size)/(1024.0*1024.0)

  fsize.append(size_in_mb)

  flabels.append(int(class_labels[class_labels["Id"] == file_name]["Class"]))

 

 file_size_df = pd.DataFrame({"Class":flabels, "Id": fname, "fsize": fsize})

 

 if not os.path.exists("./byte_file_size.csv"):

  file_size_df.to_csv("./byte_file_size.csv", index = False)

 

 print(file_size_df.head())



 print("size calculated and stored in csv")



#Call the function.

class_labels = pd.read_csv("./trainLabels.csv")

calculate_byte_file_size(class_labels)



"""



#Finally merge count vectors, file sizes and class labels. Store in a single output file, bye_count_vectors.csv
"""'

data = pd.read_csv("../input/byte_count_vectors.csv")

data = data.set_index("Id")

count_vector_data = data.drop(['Class', 'fsize'], axis=1)

count_vector_data['entropy'] = 0.0

entropies = []





for idx, rows in tqdm(count_vector_data.iterrows()):

    entropy = 0.0

    for elem in rows.index:

        if(rows[elem]):

            entropy = entropy - ( ((rows[elem]/rows.sum()) * log(rows[elem]/rows.sum())))

    

    entropies.append(entropy)



count_vector_data['entropy'] = entropies



data['entropy'] = count_vector_data['entropy']



#byte_entropy.to_csv("byte_entropy.csv")'"""
#ASM file count vectors.

#We will divide asm files in 3 different folders, 'first', 'second', 'third' respectively. 

#It will helpp us in multiprogramming.

"""

def firstprocess():

    list_of_dics = []

    files  = os.listdir("./first") 

    for file in tqdm(files):

        filename = file.split('.')[0]

        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}

        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,

                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,

                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,

                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,

                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,

                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,

                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,

                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}

        #Opcodes were changed after dchad solution.

        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}

        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,

                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}

        #Registers were changed after dchad solution.

        current_file = open("./first/"+file, "r", encoding ="cp1252", errors = "replace")

        for lines in current_file:

            line = lines.rstrip().split()

            prefix = line[0]

            rest_of_line = line[1:]        

            #Check for prefixes

            for key in prefixes.keys():

                if key in prefix:

                    prefixes[key] += 1

            #Check for keywords

            for key in keywords.keys():

                for word in rest_of_line:

                    if key in word: #Because we need to match substring.

                        keywords[key] += 1

            #Check for opcodes

            for key in opcode.keys():

                for word in rest_of_line:

                    if key==word: #Because we need to match exact string.

                        opcode[key] += 1

            #Check for registers

            if ('text' in prefix or 'CODE' in prefix):

                for key in registers.keys():

                    for word in rest_of_line:

                        if key in word:

                            registers[key] += 1

        current_file.close()

        final_dic = {'Id': filename, }

        #final_dic['Id'] = filename

        for key,values in prefixes.items():

            final_dic[key] = values

        for key,values in keywords.items():

            final_dic[key] = values

        for key,values in opcode.items():

            final_dic[key] = values

        for key,values in registers.items():

            final_dic[key] = values

        list_of_dics.append(final_dic)

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./first/firstfile.csv")

    

def secondprocess():

    list_of_dics = []

    files  = os.listdir("./second")

    for file in tqdm(files):

        filename = file.split('.')[0]

        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}

        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,

                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,

                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,

                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,

                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,

                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,

                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,

                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}

        #Opcodes were changed after dchad solution.

        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}

        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,

                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}

        #Registers were changed after dchad solution.

        current_file = open("./second/"+file, "r", encoding ="cp1252", errors = "replace")

        for lines in current_file:

            line = lines.rstrip().split()

            prefix = line[0]

            rest_of_line = line[1:]    

            #Check for prefixes

            for key in prefixes.keys():

                if key in prefix:

                    prefixes[key] += 1

            #Check for keywords

            for key in keywords.keys():

                for word in rest_of_line:

                    if key in word: #Because we need to match substring.

                        keywords[key] += 1

            #Check for opcodes

            for key in opcode.keys():

                for word in rest_of_line:

                    if key==word: #Because we need to match exact string.

                        opcode[key] += 1

            #Check for registers

            if ('text' in prefix or 'CODE' in prefix):

                for key in registers.keys():

                    for word in rest_of_line:

                        if key in word:

                            registers[key] += 1

        current_file.close()

        final_dic = {'Id': filename, }

        #final_dic['Id'] = filename

        for key,values in prefixes.items():

            final_dic[key] = values

        for key,values in keywords.items():

            final_dic[key] = values

        for key,values in opcode.items():

            final_dic[key] = values

        for key,values in registers.items():

            final_dic[key] = values

        list_of_dics.append(final_dic)

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./second/secondfile.csv")



def thirdprocess():

    list_of_dics = []

    files  = os.listdir("./third")

    for file in tqdm(files):

        filename = file.split('.')[0]

        filename = file.split('.')[0]

        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}

        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,

                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,

                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,

                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,

                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,

                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,

                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,

                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}

        #Opcodes were changed after dchad solution.

        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}

        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,

                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}

        #Registers were changed after dchad solution.

        current_file = open("./third/"+file, "r", encoding ="cp1252", errors = "replace")

        for lines in current_file:

            line = lines.rstrip().split()

            prefix = line[0]

            rest_of_line = line[1:]        

            #Check for prefixes

            for key in prefixes.keys():

                if key in prefix:

                    prefixes[key] += 1

            #Check for keywords

            for key in keywords.keys():

                for word in rest_of_line:

                    if key in word: #Because we need to match substring.

                        keywords[key] += 1

            #Check for opcodes

            for key in opcode.keys():

                for word in rest_of_line:

                    if key==word: #Because we need to match exact string.

                        opcode[key] += 1

            #Check for registers

            if ('text' in prefix or 'CODE' in prefix):

                for key in registers.keys():

                    for word in rest_of_line:

                        if key in word:

                            registers[key] += 1

        current_file.close()

        final_dic = {'Id': filename, }

        #final_dic['Id'] = filename

        for key,values in prefixes.items():

            final_dic[key] = values

        for key,values in keywords.items():

            final_dic[key] = values

        for key,values in opcode.items():

            final_dic[key] = values

        for key,values in registers.items():

            final_dic[key] = values

        list_of_dics.append(final_dic)

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./third/thirdfile.csv")

    

def main():

    p1=Process(target=firstprocess)

    p2=Process(target=secondprocess)

    p3=Process(target=thirdprocess)

    #p4=Process(target=fourthprocess)

    #p5=Process(target=fifthprocess)

    #p1.start() is used to start the thread execution

    p1.start()

    p2.start()

    p3.start()

    #p4.start()

    #p5.start()

    #After completion all the threads are joined

    p1.join()

    p2.join()

    p3.join()

    #p4.join()

    #p5.join()



if __name__=="__main__":

    main()

"""
"""

def firstprocess():

    list_of_dics = []

    files  = os.listdir("./first")

    for file in tqdm(files):

        filename = file.split('.')[0]

        current_file = open("./first/"+file, mode = "rb")

        ln = os.path.getsize("./first/"+file)

        width = int(ln**0.5)

        rem = ln%width

        a = (array.array("B"))

        a.fromfile(current_file, ln-rem)

        g = np.reshape(a, (len(a)//width, width))

        g = np.uint8(g)

        final_image_feature_array = g.flatten()[0:1000]

        current_file.close()

        keys = list("img_feature_"+ str(i) for i in range(0,1000))

        final_dic = {'Id': filename, }

        for key in keys:

            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]

        list_of_dics.append(final_dic)

	

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./first/image_feature_firstfile.csv")

    

def secondprocess():

    list_of_dics = []

    files  = os.listdir("./second")

    for file in tqdm(files):

        filename = file.split('.')[0]

        current_file = open("./second/"+file, mode = "rb")

        ln = os.path.getsize("./second/"+file)

        width = int(ln**0.5)

        rem = ln%width

        a = (array.array("B"))

        a.fromfile(current_file, ln-rem)

        g = np.reshape(a, (len(a)//width, width))

        g = np.uint8(g)

        final_image_feature_array = g.flatten()[0:1000]

        current_file.close()

        keys = list("img_feature_"+ str(i) for i in range(0,1000))

        final_dic = {'Id': filename, }

        for key in keys:

            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]

        list_of_dics.append(final_dic)

	

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./second/image_feature_secondfile.csv")



def thirdprocess():

    list_of_dics = []

    files  = os.listdir("./third")

    for file in tqdm(files):

        filename = file.split('.')[0]

        current_file = open("./third/"+file, mode = "rb")

        ln = os.path.getsize("./third/"+file)

        width = int(ln**0.5)

        rem = ln%width

        a = (array.array("B"))

        a.fromfile(current_file, ln-rem)

        g = np.reshape(a, (len(a)//width, width))

        g = np.uint8(g)

        final_image_feature_array = g.flatten()[0:1000]

        current_file.close()

        keys = list("img_feature_"+ str(i) for i in range(0,1000))

        final_dic = {'Id': filename, }

        for key in keys:

            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]

        list_of_dics.append(final_dic)

	

    first_df = pd.DataFrame(list_of_dics)

    first_df = first_df.set_index("Id")

    first_df.to_csv("./third/image_feature_thirdfile.csv")

    

def main():

    print(time.ctime())

    p1=Process(target=firstprocess)

    p2=Process(target=secondprocess)

    p3=Process(target=thirdprocess)

    #p4=Process(target=fourthprocess)

    #p5=Process(target=fifthprocess)

    #p1.start() is used to start the thread execution

    p1.start()

    p2.start()

    p3.start()

    #p4.start()

    #p5.start()

    #After completion all the threads are joined

    p1.join()

    p2.join()

    p3.join()

    print(time.ctime())

    #p4.join()

    #p5.join()



if __name__=="__main__":

    main()



"""
asm_final_features = pd.read_csv("../input/asm_final_features.csv")

asm_final_features.head()
asm_final_features = asm_final_features.set_index("Id")
asm_final_features.head()
byte_final_features = pd.read_csv("../input/byte_final_features.csv")

byte_final_features.head()