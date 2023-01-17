log = "July 31 7:51:48 mycomputer bad_process[12345]: ERROR Perfroming package upgrade"

index = log.index('[') # gives the index of i/p char 

index
# Brittle way to extracing numbers by using index function

print(log[index+1:index+6])
# re module allows for search function to find regular expressions inside strings

import re

log = "July 31 7:51:48 mycomputer bad_process[12345]: ERROR Perfroming package upgrade"

regex = r"\[(\d+)\]"

result = re.search(regex, log)

print(result[1])
# Basic Regular Expressions

# Simple Matching in Python

# The "r" at the beginning of the pattern indicates that this is a rawstring

# Always use rawstrings for regular expressions in Python

result = re.search(r'aza', 'plaza')

print(result) # strict start pos to finish+1 ; span = (=, <)
result = re.search(r'aza', 'bazaar')

print(result)
# None is a special value that Python uses that show that there's none actual value there

result = re.search(r'aza', 'maze')

print(result)
# The match attribute always has a value of the actual sub string that match the search pattern

# The span attribute indicates the range where the sub string can be found in the string

print(re.search(r"^x", "xenon"))

print(re.search(r"^e", "xenon"))

print(re.search(r"^m", "asd masd"))

print(re.search(r"^a", "asd masd"))

print(re.search(r"p.ng", "penguin"))

print(re.search(r"p.ng", "sponge"))
# Additional options to the search function can be added as a third parameter

# The re.IGNORECASE option returns a match that is case insensitive

print(re.search(r"p.ng", "Pangaea", re.IGNORECASE))

print(re.search(r"p.ng", "Pangaea"))
# Wildcards and Character Classes

# Character classes are written inside square brackets

# It list the characters to match inside of the brackets

# A range of characters can be defined using a dash

print(re.search(r"[a-z]way", "The end of the highway"))

print(re.search(r"[a-z]way", "What a way to go"))

print(re.search(r"cloud[a-zA-Z0-9]", "cloudy"))

print(re.search(r"cloud[a-zA-Z0-9]", "cloud9"))
# Use a ^, circumflex, inside the square brackets to match any characters that aren't in a group

print(re.search(r"[^a-zA-Z]", "This is a sentence with spaces."))

print(re.search(r"[^a-zA-Z ]", "This is a sentence with spaces."))
# Use a |, pipe symbol to match either one expression or another

# The search function returns the first matching string only when there are multiple matches

print(re.search(r"cat|dog", "I like cats."))

print(re.search(r"cat|dog", "I like dogs."))

print(re.search(r"cat|dog", "I like cats & dogs."))
# Use the findall function provided by the re module to get all possible matches

print(re.findall(r"cat|dog", "I like both cats and dogs."))
# Repetition Qualifiers

# Repeated matches is a common expressions that include a . followed by a *

# It matches any character repeated as many times as possible including zero - greedy behavior

print(re.search(r"Py.*n", "Pygmalion"))

print(re.search(r"Py.*n", "Python Programming"))

print(re.search(r"Py[a-z]*n", "Python Programming"))

print(re.search(r"Py[a-z]*n", "Py8hon Programming"))

print(re.search(r"Py.*n", "Pyn"))
# Use a +, plus character, to match one or more occurrences of the character that comes before it

print(re.search(r"o+l+", "goldfish"))

print(re.search(r"o+l+", "woolly"))

print(re.search(r"o+l+", "boil"))
# Use a ?, question mark symbol, for either ((ZERO or ONE)) occurrence of the character before it

# It is used to specified optional characters

print(re.search(r"p?each", "To each their own"))

print(re.search(r"p?each", "I like peaches"))
# Escaping Characters

# A pattern that includes a \ could be escaping a special regex character or a special string character

# Use a \, escape character, to match one of the special characters

print(re.search(r".com", "welcome"))

print(re.search(r"\.com", "welcome"))

print(re.search(r"\.com", "mydomain.com"))
# Use \w to match any ALPHAnumeric character including letters, numbers, and underscores

# Use \d to match DIGITS

# Use \s for matching whitespace characters like SPACE, TAB or NEWLINE

# Use \b for word boundaries

print(re.search(r"\w*", "This is an example"))

print(re.search(r"\w*", "And_this_is_another"))
# Regular Expressions in Action

# "Azerbaijan" returns "Azerbaija" because we did not specify the end 

print(re.search(r"A.*a", "Argentina"))

print(re.search(r"A.*a", "Azerbaijan"))

print("")



print(re.search(r"A*a", "Argentina"))

print(re.search(r"A*y", "Argentina"))

print("")



print(re.search(r"B*a", "Argentina"))

print(re.search(r"A*a", "AArgentina"))

print(re.search(r"B.*a", "Argentina"))
# "Azerbaijan" returns None 

# $ - should be end with prev letter

print(re.search(r"^A.*a$", "Azerbaijan"))

print(re.search(r"^A.*a$", "Australia"))
pattern = r"^[a-zA-Z0-9_]*$"

print(re.search(pattern, "this_is_a_valid_variable_name"))

print(re.search(pattern, "this isn't a valid variable name"))

print(re.search(pattern, "my_variable1"))

print(re.search(pattern, "2my_variable1"))
# Advanced Regular Expressions

# Capturing Groups

# Use parentheses to capture groups which are portions of the pattern that are enclosed in

# Below line defines two separate groups

result = re.search(r"^(\w*), (\w*)$", "Lovelace, Ada")

print(result)



# The group method returns a tuple of two elements

print(result.groups())



# Use indexing to access these groups

# The first element contains the text matched by the entire regular expression

# Each successive element contains the data that was matched by every subsequent match group

print(result[0]) # shows full group

print(result[1])

print(result[2])

print("{} {}".format(result[2], result[1]))
import re

import csv



def contains_domain(address, domain):

  """Returns True if the email address contains the given,domain,in the domain position, false if not."""

  domain = r'[\w\.-]+@'+domain+'$'

  if re.match(domain,address):

    return True

  return False





def replace_domain(address, old_domain, new_domain):

  """Replaces the old domain with the new domain in the received address."""

  old_domain_pattern = r'' + old_domain + '$'

  address = re.sub(old_domain_pattern, new_domain, address)

  return address



def main():

  """Processes the list of emails, replacing any instances of the old domain with the new domain."""

  old_domain, new_domain = 'abc.edu', 'xyz.edu'

  csv_file_location = '<csv_file_location>'

  report_file = '<path_to_home_directory>' + '/updated_user_emails.csv'

  

  user_email_list = []

  old_domain_email_list = []

  new_domain_email_list = []



  with open(csv_file_location, 'r') as f:

    user_data_list = list(csv.reader(f))

    user_email_list = [data[1].strip() for data in user_data_list[1:]]



    for email_address in user_email_list:

      if contains_domain(email_address, old_domain):

        old_domain_email_list.append(email_address)

        replaced_email = replace_domain(email_address,old_domain,new_domain)

        new_domain_email_list.append(replaced_email)



    email_key = ' ' + 'Email Address'

    email_index = user_data_list[0].index(email_key)



    for user in user_data_list[1:]:

      for old_domain, new_domain in zip(old_domain_email_list, new_domain_email_list):

        if user[email_index] == ' ' + old_domain:

          user[email_index] = ' ' + new_domain

  f.close()



  with open(report_file, 'w+') as output_file:

    writer = csv.writer(output_file)

    writer.writerows(user_data_list)

    output_file.close()



main()