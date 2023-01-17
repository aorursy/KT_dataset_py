# This code is used for loading in data from the Reddit comment database

# Don't worry about the details of this code



import sqlite3

import pandas as pd





sql_conn = sqlite3.connect('../input/reddit-comments-may-2015/database.sqlite')



comments = pd.read_sql("SELECT body FROM May2015 WHERE subreddit = 'timberwolves'", sql_conn)



comments = comments["body"]     # Convert from df to series



print(comments.shape)

comments.head(8)
comments[0].lower()      # Convert the first comment to lowercase
comments.str.lower().head(8)  # Convert all comments to lowercase
comments.str.upper().head(8)  # Convert all comments to uppercase
comments.str.len().head(8)  # Get the length of all comments
comments.str.split(" ").head(8)  # Split comments on spaces
comments.str.strip("[]").head(8)  # Strip leading and trailing brackets
comments.str.cat()[0:500]   # Check the first 500 characters
comments.str.slice(0, 10).head(8)  # Slice the first 10 characters
comments.str[0:10].head(8)  # Slice the first 10 characters
comments.str.slice_replace(5, 10, " Wolves Rule! " ).head(8)
comments.str.replace("Wolves", "Pups").head(8)
logical_index = comments.str.lower().str.contains("wigg|drew")



comments[logical_index].head(10)    # Get first 10 comments about Wiggins
len(comments[logical_index])/len(comments)
my_series = pd.Series(["will","bill","Till","still","gull"])

 

my_series.str.contains(".ill")     # Match any substring ending in ill
my_series.str.contains("[Tt]ill")   # Matches T or t followed by "ill"
"""

Regular expressions include several special character sets that allow to quickly specify certain common character types. They include:

[a-z] - match any lowercase letter 

[A-Z] - match any uppercase letter 

[0-9] - match any digit 

[a-zA-Z0-9] - match any letter or digit

Adding the "^" symbol inside the square brackets matches any characters NOT in the set:

[^a-z] - match any character that is not a lowercase letter 

[^A-Z] - match any character that is not a uppercase letter 

[^0-9] - match any character that is not a digit 

[^a-zA-Z0-9] - match any character that is not a letter or digit

Python regular expressions also include a shorthand for specifying common sequences:

\d - match any digit 

\D - match any non digit 

\w - match a word character

\W - match a non-word character 

\s - match whitespace (spaces, tabs, newlines, etc.) 

\S - match non-whitespace

"^" - outside of square brackets, the caret symbol searches for matches at the beginning of a string:

"""



ex_str1 = pd.Series(["Where did he go", "He went to the mall", "he is good"])



ex_str1.str.contains("^(He|he)") # Matches He or he at the start of a string
ex_str1.str.contains("(go)$") # Matches go at the end of a string
"""

"( )" - parentheses in regular expressions are used for grouping and to enforce the proper order of operations just like they are in math and logical expressions. In the examples above, the parentheses let us group the or expressions so that the "^" and "$" symbols operate on the entire or statement.

"*" - an asterisk matches zero or more copies of the preceding character

"?" - a question mark matches zero or 1 copy of the preceding character

"+" - a plus matches 1 more copies of the preceding character

"""





ex_str2 = pd.Series(["abdominal","b","aa","abbcc","aba"])



# Match 0 or more a's, a single b, then 1 or characters

ex_str2.str.contains("a*b.+") 
# Match 1 or more a's, an optional b, then 1 or a's

ex_str2.str.contains("a+b?a+")
"""

"{ }" - curly braces match a preceding character for a specified number of repetitions:

"{m}" - the preceding element is matched m times

"{m,}" - the preceding element is matched m times or more

"{m,n}" - the preceding element is matched between m and n times

"""



ex_str3 = pd.Series(["aabcbcb","abbb","abbaab","aabb"])



ex_str3.str.contains("a{2}b{2,}")    # Match 2 a's then 2 or more b's
ex_str4 = pd.Series(["Mr. Ed","Dr. Mario","Miss\Mrs Granger."])



ex_str4.str.contains("\. ") # Match a single period and then a space
ex_str4.str.contains(r"\\") # Match strings containing a backslash
comments.str.count(r"[Ww]olves").head(8)
comments.str.findall(r"[Ww]olves").head(8)
web_links = comments.str.contains(r"https?:")



posts_with_links = comments[web_links]



print( len(posts_with_links))



posts_with_links.head(5)
only_links = posts_with_links.str.findall(r"https?:[^ \n\)]+")



only_links.head(10)