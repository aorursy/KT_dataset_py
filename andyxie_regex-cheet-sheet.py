import re
pattern = re.compile(r'Hello')
match = pattern.match("Hello World!")
if match:
    print(match.group())
p = re.compile(r'\d+')
p.split('--1--2--3--4')
p.findall('--1--2--3--4')
numbers = p.finditer('--1--2--3--4')
for n in numbers:
    print(n.group())
p.sub("#NUMBER#", '--1--2--3--4')
Regex_Pattern = r'hackerrank'	# Do not delete 'r'.
regex_pattern = r"^(.{3}\.){3}.{3}$" # abc.def.ghi.jkx
Regex_Pattern = r"\d\d\D\d\d\D\d\d\d\d"	# \D for non digit
Regex_Pattern = r"\S\S\s\S\S\s\S\S" # \s is for white space
Regex_Pattern = r"\w\W" # \w is for word
Regex_Pattern = r'^[123][120][xs0][30Aa][xsu][\.,]$'
Regex_Pattern = r'^\D[^aeiou][^bcDF]\S[^AEIOU][^\.,]$'	# Do not delete 'r'.
Regex_Pattern = r'^[[a-z][1-9][^a-z][^A-Z][A-Z].*'