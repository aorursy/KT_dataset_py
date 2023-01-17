import requests

from pprint import pprint
access_token = "REDACTED"    # as a string (in quotes)

print("Access token: " + access_token)



# Validate access token

r = requests.get("https://www.geni.com/platform/oauth/validate_token?access_token=" + access_token).text

print(r)
def isEnglish(string):

    try:

        string.encode(encoding='utf-8').decode('ascii')

    except UnicodeDecodeError:

        return False

    else:

        return True



def normalCase(name):

    if isEnglish(name) == False:

        return name

    if len(name) < 2:

        name = name.upper()

    else:

        name = name[0].upper() + name[1:].lower()

        for i in range(len(name) - 1):

            if name[i] in [" ", "-", ".", "/", "'", '"', "(", "["] and name[i + 1] != " ":

                if i + 2 == len(name):

                    name = name[:i + 1] + name[i + 1].upper()

                else:

                    name = name[:i + 1] + name[i + 1].upper() + name[i + 2:]

        for particle in ["De ", "Van ", "Von ", "Of ", "Or ", "Et ", "And "]:

            l = len(particle)

            n = name.find(particle)

            if n != -1 and n + l < len(name):

                name = name[:n] + name[n].lower() + name[n + 1:]

        for prefix in ["Mc", "Mac", "O'", "Fitz"]:

            l = len(prefix)

            n = name.find(prefix)

            if n != -1 and n + l < len(name):

                name = name[:n + l] + name[n + l].upper() + name[n + l + 1:]

    return name



def stripId(string):  # get profile ID

    return(int(string[string.find("profile-") + 8:]))
class profile:

    def __init__(self, id, type="g"):  # type = "g" or ""

        url = "https://www.geni.com/api/profile-" + type + str(id) + "?access_token=" + access_token

        r = requests.get(url)

        data = r.json()

        if type == "g":

            self.guid = id

            self.id = stripId(data["id"])

        if type == "":

            self.id = id

            self.guid = int(data.get("guid", -1))



        self.fulldata = data  # raw data

        if "mugshot_urls" in data:

            data.pop("mugshot_urls")

        if "photo_urls" in data:

            data.pop("photo_urls")

        self.data = data

        

    def nameLifespan(self):

        birth = self.data.get("birth", {}).get("date", {}).get("year", "?")

        death = self.data.get("death", {}).get("date", {}).get("year", "?")

        if self.data.get("is_alive", "") == True:

            death = ""

        if birth == "?" and death == "?":

            return self.data.get("name", "")

        else:

            return self.data.get("name", "") + " (" + str(birth) + "－" + str(death) + ")"



    def family(self):

        url = "https://www.geni.com/api/profile-" + str(self.id) + "/immediate-family?fields=name&access_token=" + access_token

        r = requests.get(url)

        results = {}

        for key in r.json().get("nodes", {}):

            if "profile-" in key and stripId(key) != self.id:

                results[key] = r.json().get("nodes", {}).get(key)

        return results



    def parent(self, gender="male", type="birth"):

        unions = self.data.get("unions")

        for union in unions:

            url = union + "?access_token=" + access_token

            r = requests.get(url).json()

            if (type == "birth" and self.data["url"] in r.get("children", []) and self.data["url"] not in r.get("adopted_children", [])) or (type == "adopted" and self.data["url"] in r.get("adopted_children", [])):

                for url in r.get("partners", {}):

                    id = stripId(url)

                    parent = profile(id, "")

                    if parent.data["gender"] == gender:

                        return parent

        return None



    def father(self, type="birth"):

        return self.parent("male", type=type)



    def mother(self, type="birth"):

        return self.parent("female", type=type)



    def ancestor(self, generation, gender="male", type="birth"):

        p = self

        for i in range(generation):

            p = p.parent(gender=gender, type=type)

            if p == None:

                return None

        return p



    def ancestry(self, forest=[], gender="male", type="birth"):

        p = self

        lineage = {"id": p.id, "name": p.nameLifespan(), "offs": []}

        print("Getting the ancestry of:", p.nameLifespan())

        i = 1

        while addAncestorToForest(lineage, forest) == False:

            p = p.parent(gender=gender, type=type)

            if p == None:

                forest.append(lineage)

                return forest

            else:

                print("G" + str(i) + ": " + p.nameLifespan())

                lineage = {"id": p.id, "name": p.nameLifespan(), "offs": [lineage]}

            i = i + 1

        return p.data.get("id", "no name")

    

    def fix(p, indent=0):  # customized fix

        # language = "zh-TW"

        # fn = p.data.get("names", {}).get(language, {}).get("first_name", "")

        # ln = p.data.get("names", {}).get(language, {}).get("last_name", "")

        # mn = p.data.get("names", {}).get(language, {}).get("maiden_name", "")

        # dn = p.data.get("names", {}).get(language, {}).get("display_name", "")

        # title = p.data.get("names", {}).get(language, {}).get("title", "")



        fn = p.data.get("first_name", "")

        middle = p.data.get("middle_name", "")

        ln = p.data.get("last_name", "")

        dn = p.data.get("display_name", "")

        mn = p.data.get("maiden_name", "")

        suffix = p.data.get("suffix", "")

        title = p.data.get("title", "")

        if suffix in ["Rev.", "Dr."] and title == "":

            title = suffix

            suffix = ""



        url = "https://www.geni.com/api/profile-" + str(p.id) + "/update?access_token=" + access_token

        data = dict(first_name=normalCase(fn),

                    middle_name=normalCase(middle),

                    last_name=normalCase(ln),

                    maiden_name=normalCase(mn),

                    display_name=normalCase(dn),

                    title=title,

                    suffix=suffix

                    # "names": {

                    # # "en-US": dict(

                    # #             display_name=dn),

                    # language: dict(

                    #     first_name=fn[1:],

                    #     last_name=fn[0],

                    #     maiden_name=ln

                    #     # display_name="")}

                    )

        r = requests.post(url, json=data)

        print("   " * indent + "fixing", r.json().get("name", ""), r.json().get("id", "No id"))
level_max = 500



def recursion(focus, fixed=[], level=0):

    if len(fixed) > 10000 or level > level_max:

        return fixed

    print("   " * level + "Depth =", level, "| Total # of profiles =", len(fixed))

    print("   " * level + "Focus:", focus.nameLifespan())

    family = focus.family()  # a dict

    first_pass = [profile(stripId(key), "") for key in family if stripId(key) not in fixed and family[key].get("name", "") != normalCase(family[key].get("name", ""))]  # criteria

    second_pass = []  # criteria that need to access profile

    for p in first_pass:

        # fn = p.data.get("names", {}).get("zh-TW", {}).get("first_name", "")

        # ln = p.data.get("names", {}).get("zh-TW", {}).get("last_name", "")

        # suffix = p.data.get("names", {}).get("zh-TW", {}).get("suffix", "")

        # creator = p.data.get("creator", "")

        # if creator in [

        #         "https://www.geni.com/api/user-5239929",

        #         "https://www.geni.com/api/user-4730491",

        #         "https://www.geni.com/api/user-1482071"]:  # criterion

        p.fix(indent=level)

        fixed.append(p.id)

        second_pass.append(p)

    for p in second_pass:

        fixed = recursion(p, fixed, level + 1)

    return fixed





def fixCases(id, type="g"):

    p = profile(id, type)

    p.fix()

    fixed = recursion(p, fixed=[p.id])

    print("Done! Total # of Profiles =", len(fixed))

    return fixed
def makeForest(profiles):

    forest = []

    for p in profiles:

        p.ancestry(forest)

    return forest



def addAncestorToForest(ancestor, forest):  # find if ancestor is in forest, and attach

    found = False

    i = 0

    while found == False and i < len(forest):

        tree = forest[i]

        if tree.get("id") == ancestor.get("id"):

            print("attaching " + ancestor.get("name"))

            tree.get("offs").extend(ancestor.get("offs"))

            found = True

        else:

            found = addAncestorToForest(ancestor, tree.get("offs"))

        i = i + 1

    return found
def project(id, max=2200):  # just the ids, into a list

    url = "https://www.geni.com/api/project-" + str(id) + "/profiles?fields=id,name,last_name,maiden_name&access_token=" + access_token

    print("Reading: ", url)

    r = requests.get(url).json()

    data = r.get("results")

    while r.get("next_page") != None and len(data) < max:

        print("Reading: ", r["next_page"])

        url = r["next_page"] + "&access_token=" + access_token

        r = requests.get(url).json()

        data = data + r["results"]

    return data



def countProjects(data):

    counts = {}

    for result in data:

        id = stripId(result["id"])

        p = profile(id, "")

        for project in p.data.get("project_ids", []):

            if project in counts:

                counts[project] += 1

            else:

                counts[project] = 1

    return counts



def search(name, birthyear=0, deathyear=0):

    url = "https://www.geni.com/api/profile/search?names=" + name

    matches = []

    pagecount = 0

    while url != None and len(matches) < 5 and pagecount < 10:

        url = url + "&fields=id,name,birth,death&access_token=" + access_token

        r = requests.get(url)

        results = r.json().get("results", [])

        for person in results:

            if birthyear == 0 or person.get("birth", {}).get("date", {}).get("year") == birthyear:

                if deathyear == 0 or person.get("death", {}).get("date", {}).get("year") == deathyear:

                    matches.append(person)

        url = r.json().get("next_page")

        pagecount += 1

    for match in matches:

        match["id"] = stripId(match["id"])

    return matches
