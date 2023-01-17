"""

BulletinParser: Framework to standardize the job descriptions present in text format 

to structured object. Uses reusable functions and no external library 



Challenge: Data Science for Good - City of Los Angeles

Kaggle: https://www.kaggle.com/c/data-science-for-good-city-of-los-angeles

Kernel: https://www.kaggle.com/shivamb/1-bulletin-structuring-engine-cola/ 



__author__  == 'shivam bansal'

__email__   == 'shivam5992@gmail.com'

__version__ == "1.62"

"""



import re, string, os



class BulletinParser:



    """

    python class to standardize the job descriptions present in text format 

    to structured object, the final results are produced in a csv file format

    

    inputs  : the input can be a single file, or a complete folder

    outputs : single json object for one file, structured csv file for a folder

    """

    

    def __init__(self, config):

        

        """ config : root path of the file and name of the file """

        self.filename, self.path = config['filename'], config['path']      

        with open(self.path + self.filename, errors='ignore') as file_object:

            self.lines = file_object.readlines()

            self.text = "\n".join(self.lines)

            

        """ a global list of all pointers used in bulletins """

        self.chrs = "abcdefghij"

        self.nums = "0123456789"

        self.sybl = "-*"

        

        """ lookup dictionary for numbers text and values """

        self._ns = {"one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6", 

        "seven":"7", "eight":"8", "nine":"9", "ten":"10", "twelve":"12", "fourteen":"14",

        "fifteen":"15", 'sixteen':'16', "eighteen":"18", "twenty":"20","thirty-two" : "32",

        "twenty-three":"23","twenty-four":"24","twenty-seven":"27","thirty":"30",

        "thirty-six":"36", "fourty-five":"45", "sixty":"60", "ninety":"90", "135":"135"}



        """ lookup dictionaries that contain key-value pair (keywords) for identification """

        self.lookups  = {

        "full_part" : {"full time":["full time","full-time"],

                       "part time":["part time","part-time"]},

        "education" : ["college or university","education","school", "semester","quarter", 

                       "degree","coursework"],

        "experience": ["full time", "full-time", "experience", "part time", "part-time"],

        "semester"  : ["semester", "quarter", 'course'],

        "exam_type" : ["OPEN", "INT_DEPT_PROM", "DEPT_PROM", "OPEN_INT_PROM"],

        "major"     : ['degree in ', 'major in ', 'majoring'],

        "exp_flags" : [" as a ", " at the level of ", " as an "],

        "school" : {"college or university":["college or university","university or college"], 

        "college" : ["college"],"high school":["high school"],"trade school":['trade school'], 

        "apprenticeship": ["apprenticeship"],"law school" : ['law school'], 

        "technical school":['technical school']}}

        

        """ common splitters used in text parsing and cleaning in different functions """

        self.split1, self.split2 = ["\n", "and", ";"], ['\n', ';', '. ']

        self.split3 = ["; "," with "," and "," or ",". "," in "," issued ","attached","whose"]

        self.split4 = ['from an', 'from a', ' may ', ' for ', '; and']

        self.split5 = ["is required","approved","required",". ","may","with","prior"]        

        self.split5 += ["upon", ";"]

        

        """ keywords required to clean / correct some entities """

        self.spellings = { "COMPETITVE" : "COMPETITIVE", "PARMENTAL" : "PARTMENTAL"}

        spells = {"CAMPUS INTERVIEWS ONLY" : "", "MINIMUM REQUIREMENTS" : "REQUIREMENTS",

                  "REQUIREMENTS\n\n\n\nFive" : "REQUIREMENTS\n0.Five years"}

        

        """ file complete text and non empty lines """

        for k,v in spells.items():

            self.text = self.text.replace(k, v)

        self.lines = [l for l in self.text.split("\n") if l.strip()]

        

        

        

    """ utility function to find a portion of text using an identifier keyword """

    def portion(self, flag, txt=None, limit=10, next_word=True, indx=False):

        """

        a generic python function to obtain the entites using an identifier flag 

        this function is used to extract class code, requirements, duties, date.



        :params:

        input : flag keyword (example - requirement, notes, duties), limit (optional) specify 

        number of characters to be extracted around the flag, next_word : True if only one 

        (immediate) word is output, indx : flag which identifies index of the relevant lines

        output: the relevant portion related to the input flag keyword

        """

        

        """ create the list of lines variable """

        entity = ""

        lines = self.lines

        if txt != None:            

            lines = txt.split("\n")

        

        """ filter the lines which contain the identifier flag """

        lines = [" ".join(line.split()).lower() for line in lines]

        lines = [line for line in lines if flag in line]

        

        """ find relevant index if indx is set True """

        if indx:

            lines=[i for i,l in enumerate(self.lines) 

                   if l.strip().lower().startswith(flag.lower())]

        if len(lines) == 0:

            return entity 

        

        """ find the required portion of text """

        start_ix = lines[0]    

        if indx:

            end = [i for i,l in enumerate(self.lines[start_ix+1:]) if l.isupper()][0]

            entity = "\n".join(self.lines[start_ix+1:start_ix+end+1])

        else:

            """ obtain the entity text till next words (limit parameter) """

            index = start_ix.find(flag)        

            entity = start_ix[index:].replace(flag, "")

            entity = entity.strip()[:limit]

            if next_word:

                entity = entity.split()[0]    

            else:

                for split in self.split2:

                    entity = entity.split(split)[0]  

        return entity

    

    """ similar function as portion, but it executes with multiple identifiers """

    def portions(self, flag, txt, limit=10, next_word=True):

        """

        this function accept same parameters as portion, flags is a list.

        used to extract major / degree related entities from the bulletins

        """

        

        entities = []

        for flag in self.lookups[flag]:

            entity = self.portion(flag, txt, limit, next_word)

            if entity:

                entities.append(entity)

        return ", ".join(entities)



    """ utility function to check the presence of an identifier and related keywords """

    def presence_check(self, identifier, txt):

        """

        checks if a certain keyword or a list of keywords are present in the text, the function 

        performs exact and compact match, can be extend to perform more flexible text matching 

 

        :params:

        input  : identifier and the relevant portion of the text to be searched

        output : a string of identifiers which were found in the text 

        """

        

        """ obtain the list/dict of relevant tokens from the lookup dictionary """

        lookup = self.lookups[identifier] 

        

        """ iterate in lookup tokens and check if present """

        entities = []

        if type(lookup) == dict:    

            """ iterate key-value pairs, check if any value is present in the text """

            for key, values in lookup.items():    

                for val in values:

                    if val.lower() in txt.lower():

                        entities.append(key)

        else: 

            """ iterate in list, check if any element is present in the text """

            for val in lookup:

                if val.lower() in txt.lower():

                    entities.append(val)

        

        """ remove duplicates and join them by pipe """

        entities = "|".join(list(set(entities)))

        return entities

    

    """ utility function to standardize the numbers text into a numerical value """

    def standardize_numbers(self, entity):

        """

        it uses _ns lookup dictionary which defines the standardized form of a number



        :params:

        the input is uncleaned text which is probably about a number 

        the output is a numerical value

        """

        

        number = ""

        if entity.lower() in self._ns:

            number = self._ns[entity.lower()]

        elif entity in list(self._ns.values()):

            number = entity

        return number



    """ utility function to get the years/months associated with an identifer """

    def year_month(self, flag, req_text, span = 'year'):

        """

        :params:

        flag: key which is used to obtain the relevant values from the lookup dictionary 

        span: number context to be extracted, (year, month, semester, quarter)

        output: years/month associated in the text; months are converted into floats.

        """

        

        """ obtain the list of related keywords from the lookup dictionary """

        lookup = self.lookups[flag]

        

        """ iterate by lines, check if the span-value is present """

        collected = []

        for line in req_text.split("\n"):

            line = line.replace("one-year", "one year")

            """ find index of all occurances of the span in the line """

            indexes = [m.start() for m in re.finditer(span, line)]            

            

            """ slice a portion around the index within lower and upper bound """

            lower_bound, upper_bound = 30, 40

            for index in indexes:

                if index-lower_bound < 0:

                    portion = line[:index+upper_bound].lower()

                else:

                    portion = line[index-lower_bound:index+upper_bound].lower()

                

                """ next, identify if the portion of text is relevant of not, """

                """ four main conditions to identify not not important portion """

                is_relevant = False

                for keyword in lookup:                    

                    """ cond1 : keyword is not present in relevant portion """

                    if keyword not in portion:

                        continue

                    

                    """ cond2 : span is mentioned after the lookup keyword """

                    yr_ix = portion.find(span)

                    idf_ix = portion.find(keyword)

                    if yr_ix > idf_ix:

                        continue

                    

                    """ cond3 : presence of ignore words in the text portion """

                    ignore_words = ["=", "equal", "equivalent", 'lack', 'valent', 'ent to ']

                    if any(eq in portion for eq in ignore_words):

                        continue

                    

                    """ cond4 : presence of substitute in experience text """

                    if keyword == "experience":

                        if "titute" in portion:

                            continue

                    

                    """ for other cases, the portion is relevant """

                    is_relevant = True

                

                """ if relevant, then identify the numerical span value """

                if is_relevant:

                    special_checks = ["two or four", "two-year or four-year"]

                    if any(two_four in portion for two_four in special_checks):

                        collected.append("4")

                    if "two and one-half" in portion:

                        collected.append("2.5")

                    else:

                        obtained = False

                        

                        """ check entities with two words: (ex - twenty-four etc.)"""

                        for k,v in self._ns.items():

                            if "-" in k:

                                if k in portion.split(span)[0].replace(" ", "-"):

                                    collected.append(v)

                                    obtained = True

                        

                        """ for other cases, obtain the immediate previous word """

                        if obtained == False:

                            val = portion[:yr_ix].split()[-1]

                            val = val.replace("-","").replace("(","").replace(")","")

                            val = self.standardize_numbers(val)

                            if val != "":

                                collected.append(val)



        """ return the deduplicated list of month / year """

        collected = list(set(collected))

        if span == 'month':

            collected = [str(round(float(_)/12, 2)) for _ in collected]

        if len(collected) > 0:

            collected = [float(_) for _ in collected]

        return collected



    """ Custom function to identify exam type required in given job """

    def exam_type(self):

        """

        identify and cleans the required text; tags the exam type according to rules

        possible outputs for this function are open, open_int, int_dept, and dept 

        """

        

        """ Meta data related to exam types """

        bad_entities = ['an', 'on', 'both', 'only', 'a', 'to', 'nvvc', 'basis']

        bad_entities = [" "+_.upper()+" " for _ in bad_entities]

        

        """ identify and clean the main portion related to exam type """

        portion = ""

        for i, line in enumerate(self.lines):

            """ identify the relevant line """

            if "THIS EXAM" in line:

                portion = line + " " + self.lines[i+1]

                portion = " "+portion.split("GIVEN")[1]

                portion = portion.split("The City")[0].strip()+" "

                

                """ cleanup bad entities and spelling mistakes """

                for ent in bad_entities:

                    portion = portion.replace(ent, " ")

                for k,v in self.spellings.items():

                    portion = portion.replace(k,v)

                portion = " ".join(portion.split()).split(" AND ")              

                break

                

        """ join and further standardize the exam type """

        cons = ['ONLY ', 'ON ', 'BOTH ']

        exam_t = " AND ".join(portion)

        exam_t = " ".join(w for w in exam_t.split() if w not in cons)

        exam_t = exam_t.lower()

        

        """ generate the final tag of exam type """

        tag = "OPEN"

        if "open" in exam_t and "interdepartmental" in exam_t:

            tag = "OPEN_INT_PROM"

        elif "interdepartmental" in exam_t:

            tag = "INT_DEPT_PROM"

        elif "department" in exam_t:

            tag = "DEPT_PROM"

        return tag

    

    """ Custom function to obtain course counts required for the job """

    def course_count(self, req_text, flag='course', limit=50):

        """

        :params:

        req_text: complete requirement text and output: number of courses required

        output: number of courses required as the minimum requirement of the job

        """



        """ find the numerical / textual span values in the text """

        spans = list(self._ns.keys()) + list(self._ns.values())

        idx = [m.start() for m in re.finditer(flag, req_text.lower())]

        

        """ iterate and check if relevant """

        collect = []

        for each in idx:

            if each-limit < 0:

                lines = req_text[:each+limit].split("\n")

            else:

                lines = req_text[each-limit:each+limit].split("\n")

            

            """ check which spans are mentioned in the text and store """

            for l in [l for l in lines if flag in l]: 

                for span in spans:

                    if span+" "+flag in l:       

                        if span in self._ns:

                            span = self._ns[span]

                        collect.append(int(span))



        """ return the obtained value """

        if len(collect) == 0:

            return ""

        return max(collect)



    """ Custom function to obtain salary amount and DWP salary amount """

    def salary_details(self):

        """

        Identifies the salary amount mentioned in the text; also finds for DWP 

        for multiple salary amounts, only first salary is given as output

        """



        """ first identify the relevant portion """

        identifier, next_chars = "SALARY", 250

        ix = self.text.find(identifier)

        portion_x = self.text[ix:ix+next_chars]

        portion_x = portion_x.replace(identifier, "").strip()

        

        """ find the salary amount """

        salary = portion_x

        for split in self.split1:

            salary = salary.split(split)[0]

        salary = "$" + "$".join(salary.split("$")[1:])

        salary = salary.split("(")[0].split(", $")[0]

        salary = salary.split("The")[0].split(". Some")[0]

        if salary.strip() == "$":

            salary = ""

        

        """ find the DWP salary """

        dwp = ""

        rep = ["(", "flat", "$ "]

        identifier = "Department of Water and Power is "

        for line in portion_x.split("\n"):

            if identifier.lower() in line.lower():

                dwp = line.lower().split(identifier.lower())[1]

                """ basic cleanup """

                for split in self.split2 + [". "]:

                    dwp = dwp.split(split)[0]

                for r in rep:

                    dwp = dwp.replace(r, "").rstrip(".")

                dwp = dwp.replace("-rated)","").replace("-rated","")

                dwp = dwp.rstrip(".").replace("rated)","").replace("at","")

                dwp = dwp.split("and")[0].strip()

        return salary, dwp

    

    """ custom function to obtain the experience title """

    def experience_title(self, req_text):

        """

        function to identify the experience title from the requirement text 

        input is only the requirement text of a job bulletin

        """



        exp_title = []

        possed = [] 

        """ iterate in experience flags """

        for identifier in self.lookups['exp_flags']:

            for i,line in enumerate(req_text.split("\n")):

                

                """ clean and collect the relevant portions """

                if identifier in line:

                    lx = line.split(identifier)[1]

                    poss = lx

                    possed.append(poss)

                    for spliter in self.split3:

                        if spliter in [" or ", " and "]:

                            """ special check for small lines """

                            if spliter in lx and len(lx) < 60:

                                pass

                            else:

                                lx = lx.split(spliter)[0]

                        else:

                            lx = lx.split(spliter)[0]

                    exp_title.append(lx)

                    break

        exp_title = "; ".join(exp_title)

        return exp_title

    

    """ Custom function to obtain class title from the job-description text """

    def class_title(self):

        """

        extracts and cleans the job title from the bulletin text

        """

        

        entity = self.lines[0].strip().lower()

        if "revised" in entity:

            entity = entity.replace("revised", "")

        if "class code" in entity:

            entity = entity.split("class code")[0]

        return entity.title()

    

    """ custom function to identify the course subjects """

    def course_subjects(self, req_text):

        """

        this function identifes the relevant course subjects using a list of keywords 

        """



        kws = self.lookups["semester"]



        """ obtain the relevant lines """

        lines = req_text.split("\n")

        relevant_lines = [l for l in lines if any(wd in l for wd in kws)]

        relevant_lines = [l.lower() for l in relevant_lines]



        """ iterate and locate the position of relevant text """

        courses = []

        for each in relevant_lines:

            course = ""

            if all(k in each for k in kws):

                ix = each.find("quarter")

            elif "quarter" in each:

                ix = each.find("quarter")

            else:

                ix = each.find("semester")



            """ using the obtained index, split the text """

            relt = each[ix:ix+300]

            if " units in " in relt:

                course = relt.split(" units in ")[1]

                course = course.split(";")[0].split(" at ")[0]

            elif " courses in " in relt:

                course = relt.split(" courses in ")[1]

                course = course.split(";")[0].split(" at ")[0]

            elif " units of " in relt:

                course = relt.split(" units of ")[1]

                course = course.split(";")[0].split(" at ")[0]

            elif " units " in relt:

                if " in " in relt:

                    right = relt.split(" in ")[1]                    

                    if right.startswith("requ") or right.startswith("which"):

                        continue

                    if right.startswith("the"):                    

                        right = right.replace("the production", "production")

                        right = right.replace("the areas of", "")

                    if right.startswith("the"):

                        continue

                    course = right

            else:

                """ perform cleanup """

                for ent in ['university, in', 'or trade school']:

                    relt = relt.replace(ent, "")



                if "university in " in relt:

                    course = relt.split("university in ")[1].split(";")[0]

                    course = course.split(" at ")[0]



            """ more custom cleanup """

            for split in self.split4:

                course = course.split(split)[0]

            if course:

                courses.append(course)

        courses = list(set(courses))

        courses = " ".join(courses)

        

        """ further split by <num>_year """

        splitters = ['one', 'two', 'three', 'four', 'five', 'including']

        for i,split in enumerate(splitters):

            if i < 5:

                split = split + " year"

            courses = courses.split(split)[0]    

        if "coursework" in courses:

            courses = courses.split("coursework")[0]

        if courses.strip() == "it":

            courses = "IT"

        courses = courses.replace("the fields listed", "")

        courses = courses.replace("an accredited college or university.","")

        courses = courses.split("or 12 ")[0].replace("the required coursework.","")

        courses = courses.replace("any of the following areas: ","")

        return courses



    """ function to check the driver license tag """

    def driver_license(self, limit = 50):

        """

        this function finds if driving license is required or not

        

        :params:

        limit: used to define the lower and upper bound set the text portion 

        """

        

        """ find the index where driver keyword is mentioned """

        i = re.finditer("driver", self.text.lower())

        i = [m.start() for m in i]

        

        """ iterate and further check if 'license' is mentioned """

        tags = []

        for each in i:

            por = self.text[each-limit:each+limit*2].lower()

            tag = ""

            if "license" in por:

                tag = "R"

                if " may " in por:

                    tag = "P"

            tags.append(tag)

        if len(tags) > 0:

            return tags[0]

        return ""



    """  function to fix breaks in requirments in which a line is broken """

    def fix_requirement_line_breaks(self, content):

        """

        Sometimes the requirement text's flat lines are broken due to inaccuracy of 

        pdf-parsing. This function is used to fix such cases before parsing the text

        """



        remove = []

        points = ["a.","b.","c.","d.","e.", "1.", "2.", "3.", "4.", "5."]

        lines = content.split("\n")

        """ first iterate and identify the lines having no pointer """

        for i,x in enumerate(lines):

            x = x.strip()

            if not any(x[:2] == p for p in points):

                remove.append(i - 1)



        """ identfiy the index of rows and join between consecutive lines """

        rremove = [i+1 for i in remove]

        new_lines = []

        covered = []

        for j,x in enumerate(lines):

            x = x.strip()

            if j in covered:

                continue

            """ consecutive rows join operation """

            if j in remove:

                if any(x[:2] == p for p in points):

                    extra = lines[j+1].strip()

                    if extra not in x:

                        u_line = x + " " + extra

                        new_lines.append (u_line)

                        covered.append(j+1)

            else:

                new_lines.append(x)

        return "\n".join(new_lines)



    

    """ function to properly parse requirement set and subsets """

    def requirement_IDs(self, req_text):

        """

        this function extracts out the pointer associated with ever requirment

        1, 2, 3 .. are set_id, and set_text is the corresponding text 

        a, b, c .. are subset_id, and subset_text is the corresponding text

        special symbols such as * or - are also tracked; the function formats the text 

        to generate required format; Also it identifies special cases with no pointers

        """

        

        """ special case 1: breakage in requirement lines pointers """

        obtained = []

        exclude = ['POST-Certified Candidates', 'Under the policy']

        for i, l1 in enumerate(req_text.split("\n")):

            l1 = l1.strip()

            if not l1:

                continue

            if any(l1.startswith(_) for _ in exclude):

                break

            if l1[0].lower() in "12345678abcdefghi":

                obtained.append(i)

        diff = [x - obtained[i - 1] for i, x in enumerate(obtained)][1:]

        if 1 in diff:

            diff.remove(1)

        if any(x in diff for x in [2, 3, 4, 5, 6, 7, 8]):

            for i in range(8):

                content = self.fix_requirement_line_breaks(req_text)

            lines = content.split("\n")

        

        """ step2: obtain the requirement text lines """

        lines_ = [l.strip().rstrip(",") for l in req_text.split("\n")]

        

        """ step3: custom break condition """

        lines = []

        for l in lines_:

            if "substitute" in l:

                if l[0] not in ["1","2","3","4","5"]:

                    break

            elif l.startswith(exclude[0]):

                break

            lines.append(l)

                    

        """ sepcial case 2: no bullets  """

        flats = [l for l in lines if not any(l[0]==w and l[1]!="," for w in self.nums)]

        if len(lines) != 1 and len(lines) == len(flats):

            lines = [" ".join(lines)]



        """ special case 3: single line requirement """

        ignore = False

        if len(lines) == 1:

            """ check the presense of numbers or alphabets """

            for x in self.nums+self.chrs:

                starter = req_text[:2].strip().lower()

                if any(patt == starter for patt in [x+".", x+")", "("+x]):

                    ignore = True

                    

            """ check the presense of symbols """

            for x in self.sybl:

                starter = l[:1].strip().lower()

                if x == starter:

                    ignore = True

                    

            """ retun single line requirement, if of number / alphabet """

            if ignore == False:

                return ["1."], [""], [req_text], [""]

            

        

        """ for all other cases, iterate and extarct pointer """

        set_id, subset_id, set_text, subset_text  = [], [], [], []

        for i, l in enumerate(lines):

            pointer = ""

            """ check the presense of numbers or alphabets """

            for x in self.nums+self.chrs:

                starter = l[:2].strip().lower()

                if any(patt == starter for patt in [x+".", x+")", "("+x]):

                    pointer = x

                    break

                elif x == starter[0]:

                    pointer = x 

                    break

            """ check the presense of symbols """

            for x in self.sybl:

                starter = l[:1].strip().lower()

                if x == starter:

                    pointer = x

                    break

            

            """ if the pointer is obtained then slice it and the text """

            if pointer != "":             

                if pointer.rstrip(".") in self.nums:

                    sid, stext = pointer, l[2:]

                    ssid, sstext = "", ""

                else:

                    pointer = "".join(c for c in pointer if c not in ".()")

                    if pointer in self.chrs:

                        ssid, sstext = pointer, l[2:]



                """ append ids and text for set and subset """

                set_id.append(sid)

                subset_id.append(ssid)

                set_text.append(stext)

                subset_text.append(sstext)   

        

        """ following code formats the lists in the required format """

        count_dict, remove_ind = {}, []

        for i in range(len(set_text)):

            main = list(reversed(set_text))[i]

            sub = list(reversed(subset_text))[i]

            if main not in count_dict:

                count_dict[main] = 0

            count_dict[main] += 1

        for i in range(len(set_text)):

            main, sub = set_text[i], subset_text[i]

            if count_dict[main] > 1 and sub == "":

                remove_ind.append(i)

                

        """ cleanup and structure according to required format """

        uset_id, usubset_id, uset_text, usubset_text  = [], [], [], []

        for i, val in enumerate(set_id):

            if i not in remove_ind:

                uset_id.append(set_id[i])

                usubset_id.append(subset_id[i])

                uset_text.append(set_text[i])

                usubset_text.append(subset_text[i])   

        

        """ finally return the set_id, sub_ids along with their text """

        return uset_id, usubset_id, uset_text, usubset_text

    

    """ function to check the license type required """

    def license_type(self, flags = ['license']):

        """

            this function serves two purposes : identify the type of license required, 

            and other licenses required for a particular job role. possible - A, B, C

        """

        

        """ iterate and find the portion where identifier is present """

        ltypes, DL_type = [], []

        for flag in flags:

            ix = re.finditer(flag, self.text.lower())

            ix = [m.start() for m in ix]

            

            """ for search results check presense of driver keyword """

            is_dl = False

            for i in ix:

                lines = self.text[i-50:i+100].split("\n")

                line = " ".join([l for l in lines if flag in l.lower()])

                

                """ identify if driving license """

                if "driver" in line.lower():

                    is_dl = True

                if not line.strip():

                    continue

                    

                """ identify the license class """

                if "valid" in line:

                    iv = line.find("valid")

                    lx = line[iv:].replace("valid ","").split("issued")[0]

                    for split in self.split5:

                        lx = lx.split(split)[0]

                    ltype = lx

                else:

                    words = line.split(flag)[0].split(flag.title())[0]

                    words = words.split(flag.upper())[0].split()

                    up_words = [w for w in words if w[0].isupper()]

                    if len(up_words) == 0:

                        continue

                    

                    """ basic cleaning of the relevant text """

                    types = []

                    for x in reversed(words):

                        if x.islower():

                            break

                        types.append(x)

                    types = " ".join(reversed(types))

                    

                    """ replace noisy entity """

                    for r in ['1. ', '2. ', '3. ']:

                        types = types.replace(r, "").strip()

                    if types in ["A", "B"]:

                        types = ""

                    ltype = types

                

                """ save the results - DL or other licenses """

                if is_dl == True:

                    if ltype not in DL_type:

                        DL_type.append(ltype)

                else:

                    if ltype not in ltypes:

                        ltypes.append(ltype)

        

        """ deduplicate the obtained results """

        ltypes = list(set([x for x in ltypes if x.strip()]))

        ltypes = [_.strip() for _ in ltypes]

        

        """ identify the class of the driving license """

        dl_type = []

        classes = ["A", "B", "C"]

        for l in DL_type:

            l1 = "".join(_ for _ in l if _ not in string.punctuation)

            for dl_class in classes:

                if " "+dl_class+" " in l1:

                    dl_type.append(dl_class)            

            if l in ltypes:

                ltypes.remove(l)

        

        """ deduplicate and combine multiple results """

        dl_type = sorted(list(set(dl_type)))

        dl_type = " OR ".join(dl_type)

        return dl_type, ltypes



    """ function to obtain the text related to alternate job class function """

    def job_class_alt_function(self, txt):

        """

        function to obtain the text related to alternate job class function. 

        """



        _ = [" or in a class", " or at the level", " or in a position", " or as a member"]

        lines = txt.split("\n")

        for line in lines:

            line = line.strip()

            if " or " not in line:

                continue

            index = None

            for connector in _:

                if connector in line:

                    index = connector

                    break

            if index != None:

                return line.split(index)[1]

        return ""

    

    """ function to extract the function of the job role """

    def get_function(self, l):

        """

        function to obtain the text related to job class function. 

        """

        

        functions = []



        """ perform basic cleanup and proper splitting """

        l = l.split(" or in a class ")[0].replace("with the City of Los Angeles","")

        l = l.replace("; or","").strip().replace(" .","")

        

        """ handle cases which contain experience keywords """

        if "experience" in l:

            

            """ handle cases which do not contain identfier of a job role """

            if not any(w in l for w in [" as a ", " at the level of ", " as an "]):

                pr = l.split("experience")[1].split(" or in ")[0]

                pr = pr.strip().lstrip(".").lstrip(",").strip().lstrip(";")

                pr = pr.strip().lstrip(",").strip()

                if pr:

                    fw = pr.split()[0]

                    if fw in ['in','at', 'with', 'within']:

                        functions.append("experience " + pr.strip())

                    elif "ing" in fw:

                        functions.append(pr)

                    else:

                        pr = pr.strip()

                        if "responsib" in pr:

                            functions.append(pr.replace("which include the ",""))

            else:

                """ handle cases which contain experience and the identifier cases """

                if " as a " in l:

                    l = l.split(" as a ")[1]

                    if len(l.split()) <= 8:

                        pass

                    elif "experience" in l:

                        pr = l.split("experience")[1].strip()

                        pr = pr.lstrip("in the").strip()

                        pr = pr.lstrip("in ").strip()

                        functions.append("experience in " + pr)

                    elif "ing " in l:

                        got = ""

                        for w in l.split():

                            if w.endswith("ing")and w[0].islower():

                                got = w

                                break

                        if got != "":

                            got = got + "" + l.split(got)[1]

                            functions.append(got)

                elif " as an " in l:

                    l = l.split(" as an ")[1]

                    if len(l.split()) <= 8:

                        pass 

                    elif "experience" in l:

                        pr = l.split("experience")[1].strip()

                        pr = pr.lstrip("in the").strip()

                        pr = pr.lstrip("in ").strip()

                        functions.append("experience in " + pr)

                    elif "ing " in l:

                        got = ""

                        for w in l.split():

                            if w.endswith("ing") and w[0].islower():

                                got = w

                                break

                        if got != "":

                            got = got + "" + l.split(got)[1]

                            functions.append(got)

                elif " at the level of " in l:

                    l = l.split(" at the level of ")[1]

                    if len(l.split()) <= 8:

                        pass # ignore

                    elif "experience" in l:

                        pr = l.split("experience")[1].strip()

                        pr = pr.lstrip("in the").strip()

                        pr = pr.lstrip("in ").strip()

                        functions.append("experience in " + pr)

                    elif "ing " in l:

                        got = ""

                        for w in l.split():

                            if w.endswith("ing") and w[0].islower():

                                got = w

                                break

                        if got != "":

                            got = got + "" + l.split(got)[1]

                            functions.append(got)

        else:

            """ alternate to experience, also handle cases for employment """

            if "employment as a" in l:

                if "position" in l:

                    functions.append(l.split("position")[1])

        

        """ find final entity """

        func = ""

        if len(functions) > 0:

            func = functions[0].strip().rstrip(" and").rstrip(" or").rstrip(":").rstrip(";")        

            words = func.split()

            if "following" in words[-3:]:

                func = ""

        return func

        

    """ function to dig deeper into requirement text and obtain if any major is missed """

    def deep_major(self, txt):

        """ function to dig deeper into requirement text and obtain if any 

            major related entity is missed from the first step """



        major = ""

        for line in txt:

            if "university in " not in line.lower():

                continue

            if "in order" in line.lower():

                continue

            if "degree" in line.lower():

                major = line.split(" in ")[1].split("related")[0]

        return major

    

    """ utility function to add up semester and quarter together """

    def add_course(self, sem, quar):

        """

        this function is part of the restructuring component, it combines semester and quarter 

        together to give a combined value of course length in the format xSyQ

        """

        

        month, quarter = "", ""

        if len(sem) > 0:

            month = str(max(sem))+"S"

        if len(quar) > 0:

            quarter = str(max(quar))+"Q"

        course_len = month + "," + quarter

        if course_len == ",":

            course_len = ""

        return course_len

    

    """ utility function to cleanup the final entities """

    def striptext(self, txt):

        return " ".join(str(txt).strip().split()).strip()    

    

    """ utility function to obtain the misc details about the course """

    def misc_details(self, txt):

        """

        parses the requirement text, identifies additional course details 

        

        :params: 

        txt: requirement text where misc details about course are captured

        """

        

        misc_details = ""

        course_lists = self.lookups['education'] + self.lookups['semester']

        ignore = ['experience', 'college', 'high school', 'trade school']

        for l in txt.split("\n"):   

            if any (w in l for w in ignore):

                continue

            if any(w in l.lower() for w in course_lists):

                misc_details += l+"\n"

        misc_details = misc_details.strip().rstrip("or").rstrip("and")

        return misc_details



    """ master function to call all the other functions and generate the output """

    def standardize(self):

        """

        master function 'standardize' is the main function which calls all the 

        other functions in order to obtain structured information

        """

        

        """ create an empty list to store the results """

        rows = []

        form = Formatter()

        

        """ first obtain the requirement text """

        requirement_text = self.portion("requirement", indx = True)  

        requirement_list = self.requirement_IDs(requirement_text)

        

        """ iterate in every requirement line, and call the other functions """

        for j in range(0, len(requirement_list[0])):            

            doc = {}

            doc['FILE_NAME'] = " ".join(self.filename.split())

            

            """ store the set-id, set-text, subset-id, subset-text """

            doc["REQUIREMENT_SET_ID"] = requirement_list[0][j]

            doc["REQUIREMENT_SET_TEXT"] = requirement_list[2][j]

            doc["REQUIREMENT_SUBSET_ID"] = requirement_list[1][j]

            doc["REQUIREMENT_SUBSET_TEXT"] = requirement_list[3][j]

            

            """ requirement conjuction """

            conjunction = []

            t1 = requirement_list[2][j].strip()

            t2 = requirement_list[3][j].strip()

            if t1.endswith("or"):

                conjunction.append("or")

            elif t1.endswith("and"):

                conjunction.append("and")

            elif t2.endswith("and"):

                conjunction.append("and")

            elif t2.endswith("or"):

                conjunction.append("or")

            doc['REQUIREMENT_CONJUNCTION'] = ";".join(conjunction).rstrip(";")

            

            """ add classcode, duties, open date using generic functions """

            doc['JOB_CLASS_NO'] = self.portion("class code:")

            doc['JOB_DUTIES'] = self.portion("duties", indx=True)        

            doc['OPEN_DATE'] = self.portion("open date:")



            """ create a combined requirement text row, to be used for other entities """

            rtext = doc["REQUIREMENT_SET_TEXT"] +"\n"+ doc['REQUIREMENT_SUBSET_TEXT']

            rtext = re.sub(r'\([^)]*\)', '', rtext)            



            """ add schooltype, experience type date using generic functions """

            doc["SCHOOL_TYPE"] = self.presence_check("school", txt=rtext)

            doc["FULL_TIME_PART_TIME"] = self.presence_check("full_part", txt=rtext)

                        

            """ custom functions that uses requirement text """

            doc["EXP_JOB_CLASS_ALT_RESP"] = self.job_class_alt_function(rtext)

            doc["EXP_JOB_CLASS_TITLE"] = self.experience_title(rtext)

            doc["COURSE_SUBJECT"] = self.course_subjects(rtext) 

            doc["COURSE_COUNT"] = self.course_count(rtext)

            doc["ENTRY_SALARY_GEN"] = self.salary_details()[0]

            doc["ENTRY_SALARY_DWP"] = self.salary_details()[1]

            doc["DRIVERS_LICENSE_REQ"] = self.driver_license()

            doc["MISC_COURSE_DETAILS"] = self.misc_details(rtext)

            doc["EXP_JOB_CLASS_FUNCTION"] = self.get_function(rtext)

            

            """ custom general functions """

            doc["DRIV_LIC_TYPE"] = self.license_type()[0]

            doc['JOB_CLASS_TITLE'] = self.class_title()

            doc["ADDTL_LIC"] = self.license_type()[1]

            doc["EXAM_TYPE"] = self.exam_type()      

                        

            """ identify year / months / number using generic functions """

            doc["Exp_y"] = self.year_month("experience", rtext, span = 'year')

            doc["Exp_m"] = self.year_month("experience", rtext, span = 'month')

            doc["Cor_s"] = self.year_month("semester", rtext, span='semester')

            doc["Cor_q"] = self.year_month("semester", rtext, span='quarter')

            doc["Edu_y"] = self.year_month("education", rtext, span = 'year')

            

            """ add course and experience length by aggregating years and months """

            doc["COURSE_LENGTH"] = self.add_course(doc['Cor_s'], doc['Cor_q'])

            doc["EDUCATION_MAJOR"] = self.portions('major',rtext,limit=180,next_word=False)

            doc["EXPERIENCE_LENGTH"], doc["EDUCATION_YEARS"]   = "", ""

            

            """ cleanup and restructuring """

            if len(doc["Exp_y"] + doc["Exp_m"]) > 0:

                doc["EXPERIENCE_LENGTH"] = max(doc["Exp_y"] + doc["Exp_m"])            

            if len(doc['Edu_y']) > 0:

                doc["EDUCATION_YEARS"] = doc["Edu_y"][0]

            if doc['EDUCATION_MAJOR'] == "":

                mix = requirement_list[2][j].split("\n")+requirement_list[3][j].split("\n")

                doc['EDUCATION_MAJOR'] = self.deep_major(mix)

            if "college or university" in doc["SCHOOL_TYPE"].lower():

                doc["SCHOOL_TYPE"] = doc["SCHOOL_TYPE"].replace("|college","")



            """ perform further cleaning and standardizing on some fields """

            exp_title, alt_exp = doc["EXP_JOB_CLASS_TITLE"], doc["EXP_JOB_CLASS_ALT_RESP"]

            doc["EXP_JOB_CLASS_ALT_RESP"] = form.cleanup(alt_exp, tag='alt_exp')

            doc["EXP_JOB_CLASS_TITLE"] = form.cleanup(exp_title, tag='exp').title()

            doc["EDUCATION_MAJOR"] = form.cleanup(doc["EDUCATION_MAJOR"], tag='major')  

            doc["ADDTL_LIC"] = form.cleanup(doc["ADDTL_LIC"], tag='add_lic')

            doc["OPEN_DATE"] = form.cleanup(doc["OPEN_DATE"], tag='date')  

            

            """ append the key-value pairs in the global list """

            rows.append({k:v for k,v in doc.items() if not k.startswith("_")})

        return rows
class Formatter:

    """

    python class to format, restructure, and clean the extracted entities from different 

    functions of the BulletinParser class. 

    """

    

    def __init__(self):

        self.split6 = [" and ", "which", " with ", ", or any", "(as",'such as', 'from ', 

        'may be', 'or a closely',  ', or upon', ', and one', ', which', ', or closely', '(as']

        self.major_reps = [', , includi', ', or other', ', or in a closely related field', 

        'or a closely related field', 'a related field',', or a', ', or a related field'] 

        self.exp_starts = ['at the ', 'at that ', 'which is at least at the ', 'which is ',

        'which is at that ', 'which is at the ', 'atleast at the ', 'at least at the ']

        self.exp_starts1 = ['level', 'atleast', 'in', 'the', 'which', 'at', 'least', 'of','as ',

                           'a ', 'that', 'with']

        self.ignore_exp = ["uniformed", "helper", "buyer", "civil", "engaged", "exempt", 

        "lead", "construction", "crime", "heating", "maintenance", "insulator"]

        self.exp_starts += ['atleast at that ', 'at least at that ']

        self.major_reps += ['the text box which will appear']

        

    """ custom function used to cleanup the text/entites obtained from different functions """

    def cleanup(self, txt, tag):

        """

        this function is used to clean the extracted according to different rules, this function

        removes the noise captured along with the entity. Custom rules are used for this purpose. 

        

        :params:

        tag: identifies what type of cleaning is required

        """

        

        cleaned = ""

        if tag == 'date':

            """ cleaning the date """

            if "-" in txt:

                txt = txt.split("-")

                m, d, y = txt[0], txt[1], txt[2]

                if len(m) == 1:

                    m = "0"+m

                if len(y) == 4:

                    y = y[2:]

                cleaned = m+"-"+d+"-"+y

                

        elif tag == 'major':

            """ cleaning the major """

            if not txt.startswith("the education section"):

                cleaned = txt.replace("major in ","")

                for split in self.split6:

                    cleaned = cleaned.split(split)[0]

                for r in self.major_reps:

                    cleaned = cleaned.replace(r, "")

                cleaned = cleaned.lstrip(",").rstrip(",").strip()

                for st in ['such as', 'a ', 'an ']:

                    if cleaned.startswith(st):

                        cleaned = cleaned.replace(st, "")

                for st in [', or', ', and']:

                    if cleaned.endswith(st):

                        cleaned = cleaned.split(st)[0]                    

                if cleaned.startswith("in "):

                    cleaned = cleaned[3:].strip()

                if cleaned.endswith(" or"):

                    cleaned = cleaned[:-3]

                cleaned = cleaned.strip().rstrip(",")

        

        elif tag == 'add_lic':

            """ cleaning the additional license required """

            cleaned = []

            for line in txt:

                line = line.lower()

                line = line.replace("'s", "")

                if line.startswith("a "):

                    line = line[2:].strip()

                if line.endswith("license"):

                    line = line.replace("license", "")

                if " as a " in line:

                    line = line.split(" as a")[1].strip()

                line = line.replace(" (pe)","")

                if any(line == x for x in ['special', 'the', 'this']):

                    line = ""

                line = line.strip() + " License"

                line = " "+line+" "

                line = line.replace(" pe ", " PE ").strip()

                if line == "License":

                    line = ""

                else:

                    cleaned.append(line)

            cleaned = ", ".join(cleaned)



        elif tag == 'exp':

            """ cleaning the experience """

            cleaned = []

            txt = txt.rstrip(",").replace("engaged ","").replace("either ", "")

            for split in ["which", "one year of", "for "]:

                txt = txt.split(split)[0]

            if len(txt.split()) == 1:

                if txt.lower().split()[0] in self.ignore_exp:

                    txt = ""        

            for i,_ in enumerate(txt.split()):

                if _[0].isupper() == True:

                    pass

                elif i == 0:

                    pass

                elif _.lower()[:5] in ["build", "plumb", "condi", "housi"]:

                    pass

                elif _.replace(",","").endswith("ing"):

                    break

                cleaned.append(_)

            cleaned = " ".join(cleaned)

            

            

            if cleaned.endswith("engaged"):

                cleaned = cleaned.replace("engaged", "")

            if "(" in cleaned:

                cleaned = cleaned.split("(")[0]

            elif " by " in cleaned:

                cleaned = cleaned.split(" by ")[0]

            elif cleaned.startswith("a "):

                cleaned = cleaned[2:].strip()

            elif cleaned.startswith("an "):

                cleaned = cleaned[3:].strip()

                

            cleaned = cleaned.rstrip(",").rstrip(".").replace(", two years of","")

            cleaned = cleaned.split("Class Code")[0]

            if " the level of " in cleaned:

                cleaned = cleaned.split(" the level of ")[1]

            cleaned = cleaned.replace(", or","").split("within")[0]

            if cleaned.lower().endswith("construction"):

                cleaned = cleaned.lower().replace("construction", "").title()

            if cleaned.endswith("responsible"):

                cleaned = cleaned.replace("responsible", "")

            cleaned = cleaned.rstrip(",").rstrip(".")

            if cleaned.strip().endswith(" of"):

                cleaned = cleaned[:-2].strip()

            if cleaned.strip().endswith(" or"):

                cleaned = cleaned[:-2].strip()

            if cleaned.strip().endswith(" and"):

                cleaned = cleaned[:-3].strip()



            ## more cleaning rules

            cleaned = cleaned.split("may be sub")[0]

            cleaned = cleaned.replace("lead; ","").replace("engaged; ","")

            if cleaned.startswith("exempt"):

                cleaned = cleaned.replace("exempt or","").replace("exempt","")

            if cleaned.startswith("building, construction,"):

                cleaned = ""

            if cleaned.startswith("housing, building, electrical"):

                cleaned = ""

            if cleaned.endswith(", building"):

                cleaned = cleaned.replace(", building","")

                



        if tag == 'alt_exp':

            """ cleaning the alternate experience """

            cleaned = txt.strip()

            if cleaned:          

                cleaned = cleaned.replace("with the City of Los Angeles;","")

                for e in self.exp_starts:

                    cleaned = cleaned.replace(e,"").strip()

                for e in self.exp_starts1:

                    if cleaned.startswith(e):

                        cleaned = cleaned[len(e):].strip()

                        cleaned = cleaned.lstrip(",").strip()

                for e in self.exp_starts1:

                    if cleaned.startswith(e):

                        cleaned = cleaned[len(e):].strip()

                        cleaned = cleaned.lstrip(",").strip()

                if cleaned.endswith("or"):

                    cleaned = cleaned[:-2]

                if cleaned.endswith("and"):

                    cleaned = cleaned[:-3]

        return cleaned
import pandas as pd 

class Extractor:

    """

    Controller Class, which executes the key piece of code required to obtain entities and 

    information for different files, and producing a structured file as output

    """

    

    def __init__(self, path):

        """

        python class to format, restructure and remove noise from obtained entities

        most of the noise is handelled during the entity extraction but sometimes there is 

        still presence of noisy keywords. 

        """

        

        self.path   = path

        self.column = ['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'REQUIREMENT_SET_ID', 

        'REQUIREMENT_SUBSET_ID', 'JOB_DUTIES', 'EDUCATION_YEARS', 'SCHOOL_TYPE', 

        'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME', 

        'EXP_JOB_CLASS_TITLE', 'EXP_JOB_CLASS_ALT_RESP', 'EXP_JOB_CLASS_FUNCTION', 

        'COURSE_COUNT', 'COURSE_LENGTH', 'COURSE_SUBJECT', 'MISC_COURSE_DETAILS', 

        'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 'EXAM_TYPE', 'ENTRY_SALARY_GEN', 

        'ENTRY_SALARY_DWP', 'OPEN_DATE', 'REQUIREMENT_SET_TEXT', 'REQUIREMENT_SUBSET_TEXT', 

        'REQUIREMENT_CONJUNCTION']



    """ function to iterate in files and generate a structured csv file """

    def extraction(self):

        """ 

            master function that iterates in the root folder for every file 

            and obtains the structured information, final output is a csv file

        """

        

        print ("> Starting Extraction for the Given Folder")

        rows = []

        files = sorted(os.listdir(path))

        for filename in files:

            try:

                config = {"path" : self.path, "filename" : filename}

                bp = BulletinParser(config)

                rows.extend(bp.standardize())

            except Exception as E:

                print (E)

                continue



        df = pd.DataFrame(rows)[self.column]

        for c in df.columns:

            df[c] = df[c].apply(lambda x : "-" if x == "" else x)

            df[c] = df[c].apply(lambda x : bp.striptext(x))

        df.to_csv("structured_file.csv", index = False)

        print (">> Extraction Complete for Entire Folder")

        return df
## define the input path

path = "../input/cityofla/CityofLA/Job Bulletins/"



## create the Extractor Class object

extr = Extractor(path)



## call the extraction function

data = extr.extraction()

data.head(10)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download Complete Structured CSV file", filename = "structured_file.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = """<h1><a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a></h1>

    

    <h3><a href='https://docs.google.com/spreadsheets/d/1MuzQd4fuKvElQvVXCG68KOLSgoo0IBTEQseaW_Z7JCE/edit?usp=sharing'>Updated Data Dictionary</a></h3>"""

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(data)