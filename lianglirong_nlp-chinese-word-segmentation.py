import numpy as np
import os
print(os.listdir("../input/nlp-data"))
#逆向最大匹配
class RMM(object):
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        #读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)
        #print("maximum:",self.maximum)
        
    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index - size):index]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break
            if word is None:
                index -= 1
        return result[::-1]
class MMM(object):
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        #读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)
                    
        #print("maximum:",self.maximum)

    def cut(self, text):
        result = []
        text_len = len(text)
        index = 0
        
        while index != text_len:
            word = None
            for size in range(self.maximum, 0, -1):
                if text_len - (size+index) < 0:
                    continue
                #print(index)
                piece = text[index:index+size]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                result.append(text[index])
                index += 1
        return result
text = "南京市长江大桥"
rmm = RMM("../input/nlp-data/imm_dic.utf8")
rmm.cut(text)
mmm = MMM("../input/nlp-data/imm_dic.utf8")
mmm.cut(text)
