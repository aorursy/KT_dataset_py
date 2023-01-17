#ratio
from fuzzywuzzy import fuzz
Str1 = "Fuzzy Pie"
Str2 = "fuzzy pie"
Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
print(Ratio)
#partial_ratio
Str1 = "Fuzzy Pie"
Str2 = "two slices of fuzzy pie."
Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
print(Ratio)
print(Partial_Ratio)
#token_sort_ratio
Str1 = "Canada vs US"
Str2 = "US vs Canada"
Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
Token_Sort_Ratio = fuzz.token_sort_ratio(Str1,Str2)
print(Ratio)
print(Partial_Ratio)
print(Token_Sort_Ratio)
#token_set_ratio
Str1 = "Trump officially withdraws US from World Health Organization"
Str2 = "US"
Ratio = fuzz.ratio(Str1.lower(),Str2.lower())
Partial_Ratio = fuzz.partial_ratio(Str1.lower(),Str2.lower())
Token_Sort_Ratio = fuzz.token_sort_ratio(Str1,Str2)
Token_Set_Ratio = fuzz.token_set_ratio(Str1,Str2)
print(Ratio)
print(Partial_Ratio)
print(Token_Sort_Ratio)
print(Token_Set_Ratio)
#process.extract/extractOne
from fuzzywuzzy import process
choices = ["Atlanta Falcons","New York Jets", "New York Giants", "Dallas Cowboys"]
multipleresults = process.extract("new york jets", choices, limit=3)
oneresult = process.extractOne("cowboys", choices)
print(multipleresults)
print(oneresult)